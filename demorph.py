
import os
import glob
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


from FM.adaface import get_adaface_embedding, get_adaface_model
from tqdm.auto import tqdm
import argparse




model = "CompVis/stable-diffusion-v1-4"
autoencoder = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse",cache_dir='./cache').cuda()
autoencoder.requires_grad_(False);


generator=UNet2DModel(in_channels=4,out_channels=8)
generator=torch.nn.DataParallel(generator)
generator.load_state_dict(torch.load('./pretrained/latent-conditional-gan.pth'))

generator=generator.module.cuda()
generator.eval();



with torch.no_grad():
    def get_batch_recon(autoencoder,generator,batch,image =False):
        morph_enc=autoencoder.encode(batch['morphed_image'].cuda()).latent_dist.mode()
        batch_={'B':morph_enc} 
        real_A = Variable(batch_["B"].type(torch.cuda.FloatTensor))
        fake_B = generator(real_A,timestep=0).sample
        fakeb1,fakeb2=fake_B[:,0:4,:,:],fake_B[:,-4:,:,:]
        recon1=autoencoder.decode(fakeb1).sample
        recon2=autoencoder.decode(fakeb2).sample
        return recon1,recon2



def process_single_morph( autoencoder, generator,opt):
    
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
    # Derive img1 and img2 paths based on naming convention
    filename = os.path.basename(opt.morph_path)


    # Load and transform
    morph = transform(Image.open(opt.morph_path).convert("RGB"))


    # Create batch
    batch = {
        'morphed_image': morph.unsqueeze(0)  # Add batch dim
    }

    # Run through autoencoder + generator
    recon1_batch, recon2_batch = get_batch_recon(autoencoder, generator, batch, image=False)

    # Convert to images for plotting and saving
    morph_img = invTrans(batch['morphed_image'][0])
    recon1_img = invTrans(recon1_batch[0])
    recon2_img = invTrans(recon2_batch[0])


    # Plot
    images = [morph_img, recon1_img, recon2_img]
    titles = ['Morph', 'OUT1', 'OUT2']

    plt.figure(figsize=(15,5))
    for j, img in enumerate(images):
        plt.subplot(1,3,j+1)
        plt.imshow(img)
        plt.title(titles[j])
        plt.axis('off')
    plt.show()

    # Save reconstructions
    recon1_img.save(os.path.join(opt.save_path, filename.replace("morph", "recon1")))
    recon2_img.save(os.path.join(opt.save_path, filename.replace("morph", "recon2")))
    print(f"Saved reconstructed images to {opt.save_path}")

# ----------------------------
# Command-line interface
# ----------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process a single morph image")
    parser.add_argument(
        "--morph_path", type=str, required=True,
        help="Path to the morph image"
    )
    parser.add_argument(
        "--save_path", type=str, required=True,
        help="Path to the save directory"
    )
    parser.add_argument("--img_height", type=int, default=512,
                        help="Image height (default: 512)")

    parser.add_argument("--img_width", type=int, default=512,
                        help="Image width (default: 512)")
    opt = parser.parse_args()

    # Example usage: you need to define your autoencoder and generator objects
    process_single_morph(autoencoder, generator, opt)