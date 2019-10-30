#%%
from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np 
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms, utils 

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
#%%
class FaceDataset(Dataset):
    """ WikiFaces Dataset"""
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir(string): Directory to all the images
            transform(callable, optional): Optional transform to be applied 
            on a sample
        """
        self.root_dir = root_dir
        self.transform = transform

    def __getitem__(self):



        if self.transform:
            sample = self.transform
            