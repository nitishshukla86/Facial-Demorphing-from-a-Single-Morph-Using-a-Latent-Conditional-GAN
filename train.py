import os
import glob
import math
import time
import datetime
import itertools
from pathlib import Path
from typing import Optional, Tuple, Union
import warnings
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

import matplotlib.pyplot as plt
import cv2
from PIL import Image
from skimage import io
from skimage.metrics import structural_similarity as sk_cpt_ssim

from diffusers import (
    UNet2DConditionModel,
    UNet2DModel,
    DiffusionPipeline,
    ImagePipelineOutput,
    DDPMScheduler,
    AutoencoderKL,
)

from accelerate import Accelerator
from huggingface_hub import HfFolder, Repository, whoami
from transformers import AutoProcessor, AutoTokenizer, CLIPModel

from facenet_pytorch import MTCNN
from deepface import DeepFace
from FM.adaface import get_adaface_embedding, get_adaface_model

from tqdm.auto import tqdm
from dataset import SMDD
from loss import KurtosisLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False


model = "CompVis/stable-diffusion-v1-4"
autoencoder = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse",cache_dir='./cache').to(device)
autoencoder.requires_grad_(False)




def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)






import argparse

def get_training_config():
    parser = argparse.ArgumentParser(description="Training configuration")

    parser.add_argument("--epoch", type=int, default=0, help="Epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=2, help="Evaluation batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for Adam optimizer")
    parser.add_argument("--b1", type=float, default=0.5, help="Adam beta1")
    parser.add_argument("--b2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--decay_epoch", type=int, default=100, help="Epoch to start LR decay")
    parser.add_argument("--n_cpu", type=int, default=1, help="Number of CPU threads for batch generation")
    parser.add_argument("--img_height", type=int, default=512, help="Image height")
    parser.add_argument("--img_width", type=int, default=512, help="Image width")
    parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
    parser.add_argument("--sample_interval", type=int, default=500, help="Interval for sampling images")
    parser.add_argument("--checkpoint_interval", type=int, default=-1, help="Interval for model checkpoints")
    parser.add_argument("--eval_interval", type=int, default=20, help="Interval for evaluation")

    args = parser.parse_args()
    args.dataset_name = "SMDD"
    
    return args


opt = get_training_config()





transform = transforms.Compose([        
        transforms.ToTensor(),
        transforms.Resize((opt.img_height,opt.img_width),antialias=False),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                                transforms.ToPILImage()
                               ])
trainset = SMDD(train=True, transform=transform)#Subset(dataset, train_indices)
testset = SMDD(train=False,transform=transform)#Subset(dataset, test_indices)
dataloader=DataLoader(trainset,batch_size=opt.batch_size,drop_last=True,shuffle=True)
val_dataloader=DataLoader(testset,batch_size=opt.eval_batch_size,drop_last=True,shuffle=True)



os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)


# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()
kurtosis=KurtosisLoss()


# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100

# Calculate output of image discriminator (PatchGAN)
patch = (1, int(opt.img_height/8) // 2 ** 4, int(opt.img_width/8) // 2 ** 4)

# Initialize generator and discriminator
# generator = GeneratorUNet()
generator=UNet2DModel(in_channels=4,out_channels=8)
discriminator = Discriminator(in_channels=4)

if cuda:
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    criterion_GAN.to(device)
    criterion_pixelwise.to(device)

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))
    discriminator.load_state_dict(torch.load("saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))



# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def sample_images(epoch,autoencoder):
    """Saves a generated sample from the validation set"""
    batch = next(iter(val_dataloader))

    morph_enc=autoencoder.encode(batch['morphed_image'].to(device)).latent_dist.mode()
    img1_enc=autoencoder.encode(batch['img1'].to(device)).latent_dist.mode()
    img2_enc=autoencoder.encode(batch['img2'].to(device)).latent_dist.mode()
    batch_={'B':morph_enc,'A':torch.cat([img1_enc,img2_enc],1)}    # Model inputs

    real_A = Variable(batch_["B"].type(Tensor))
    real_B = Variable(batch_["A"].type(Tensor))
    fake_B = generator(real_A,timestep=0).sample

    fakeb1,fakeb2=fake_B.chunk(2,1)


    recon1=autoencoder.decode(fakeb1).sample
    recon2=autoencoder.decode(fakeb2).sample
    img_sample = torch.cat((batch['morphed_image'].to('cpu'), recon1.to('cpu'),recon2.to('cpu'), batch['img1'].to('cpu'),batch['img2'].to('cpu')), -2)
    save_image(img_sample, "images/%s/%s.png" % (opt.dataset_name, epoch), nrow=5, normalize=True)


# # ----------
#  Training


from accelerate import Accelerator
accelerator = Accelerator()
generator,discriminator,autoencoder, optimizer_G ,optimizer_D, dataloader,val_dataloader = accelerator.prepare(
     generator,discriminator,autoencoder, optimizer_G ,optimizer_D, dataloader,val_dataloader )



prev_time = time.time()

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        morph_enc=autoencoder.encode(batch['morphed_image'].to(device)).latent_dist.mode()
        img1_enc=autoencoder.encode(batch['img1'].to(device)).latent_dist.mode()
        img2_enc=autoencoder.encode(batch['img2'].to(device)).latent_dist.mode()
        batch={'B':morph_enc,'A':torch.cat([img1_enc,img2_enc],1)}
        
        real_A = Variable(batch["B"].type(Tensor))
        real_B = Variable(batch["A"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # GAN loss
        fake_B = generator(real_A,timestep=0).sample
        # print(fake_B.shape, real_A.shape)
        pred_fake = discriminator(fake_B, real_A)
        # print(pred_fake.shape,valid.shape)

        loss_GAN = criterion_GAN(pred_fake, valid)
        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_B, real_B) + kurtosis.forward(fake_B)

        # Total loss
        loss_G = loss_GAN + lambda_pixel * loss_pixel

        loss_G.backward()

        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Real loss
        pred_real = discriminator(real_B, real_A)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = discriminator(fake_B.detach(), real_A)
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        if accelerator.is_main_process:
        # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_pixel.item(),
                    loss_GAN.item(),
                    time_left,
                )
            )


    if epoch % opt.eval_interval==0 and accelerator.is_main_process:
        sample_images(epoch,autoencoder)
        torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (opt.dataset_name, epoch))
