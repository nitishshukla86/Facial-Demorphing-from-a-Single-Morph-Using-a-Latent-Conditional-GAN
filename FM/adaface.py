from transformers import AutoModel
from huggingface_hub import hf_hub_download
import shutil
import os
import torch
import sys


# helpfer function to download huggingface repo and use model
def download(repo_id, path, HF_TOKEN=None):
    os.makedirs(path, exist_ok=True)
    files_path = os.path.join(path, 'files.txt')
    if not os.path.exists(files_path):
        hf_hub_download(repo_id, 'files.txt', token=HF_TOKEN, local_dir=path, local_dir_use_symlinks=False)
    with open(os.path.join(path, 'files.txt'), 'r') as f:
        files = f.read().split('\n')
    for file in [f for f in files if f] + ['config.json', 'wrapper.py', 'model.safetensors']:
        full_path = os.path.join(path, file)
        if not os.path.exists(full_path):
            hf_hub_download(repo_id, file, token=HF_TOKEN, local_dir=path, local_dir_use_symlinks=False)

            
# helpfer function to download huggingface repo and use model
def load_model_from_local_path(path, HF_TOKEN=None):
    cwd = os.getcwd()
    os.chdir(path)
    sys.path.insert(0, path)
    model = AutoModel.from_pretrained(path, trust_remote_code=True, token=HF_TOKEN)
    os.chdir(cwd)
    sys.path.pop(0)
    return model


# helpfer function to download huggingface repo and use model
def load_model_by_repo_id(repo_id, save_path, HF_TOKEN=None, force_download=False):
    if force_download:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
    download(repo_id, save_path, HF_TOKEN)
    return load_model_from_local_path(save_path, HF_TOKEN)

def get_adaface_embedding(model,img):
    # input is a rgb image normalized.
    from torchvision.transforms import Compose, ToTensor, Normalize,Resize
    from PIL import Image
    trans = Compose([ToTensor(), Resize((112,112)), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    input = trans(img).unsqueeze(0)  # torch.randn(1, 3, 112, 112)
    out = model(input)
    return out

def get_adaface_model():
    HF_TOKEN = os.getenv("HF_TOKEN")
    path = os.path.expanduser('~/.cvlface_cache/minchul/cvlface_adaface_ir101_webface12m')
    repo_id = 'minchul/cvlface_adaface_ir101_webface12m'
    model = load_model_by_repo_id(repo_id, path, HF_TOKEN)

    return model


