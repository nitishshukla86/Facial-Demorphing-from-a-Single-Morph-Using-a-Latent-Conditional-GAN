import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset
from PIL import Image

class SMDD(Dataset):
    """
    Modified SMDD Dataset

    Each sample returned by this dataset is a dictionary with the following keys:

    Keys:
        'morphed_image' : torch.Tensor or PIL.Image
            The morphed (synthetic) face image.
        'img1'          : torch.Tensor or PIL.Image
            The first real face image used to generate the morph.
        'img2'          : torch.Tensor or PIL.Image
            The second real face image used to generate the morph.

    Notes:
        - The dataset reads from 'self_made_morphs_cropped.csv', which contains the following columns:
            'morph'     : path to the morphed image
            'img1'      : path to the first real image
            'img2'      : path to the second real image
            'is_train'  : 1 for training split, 0 for test split
        - Image paths should be valid and point to cropped face images.
        - If a transform is provided, it is applied to all three images.
        - Use `is_train=True` to load the training split, `False` for the test split.

    """
    def __init__(self, train=True, transform=None):
        # Read CSV and filter train/test split
        self.df = pd.read_csv('self_made_morphs_cropped.csv')
        self.df = self.df[self.df['is_train'] == int(train)].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        row = self.df.iloc[idx]

        # Load images
        morph = cv2.imread(row['morph'])
        img1  = cv2.imread(row['img1'])
        img2  = cv2.imread(row['img2'])

        # Convert BGR to RGB
        morph = cv2.cvtColor(morph, cv2.COLOR_BGR2RGB)
        img1  = cv2.cvtColor(img1,  cv2.COLOR_BGR2RGB)
        img2  = cv2.cvtColor(img2,  cv2.COLOR_BGR2RGB)

        # Convert to PIL Image for torchvision transforms
        morph = Image.fromarray(morph)
        img1  = Image.fromarray(img1)
        img2  = Image.fromarray(img2)

        if self.transform:
            morph = self.transform(morph)
            img1  = self.transform(img1)
            img2  = self.transform(img2)

        return {
            'morphed_image': morph,
            'img1': img1,
            'img2': img2
        }
