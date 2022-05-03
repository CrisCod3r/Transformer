# Image libraries
from PIL import Image 
import cv2

# Path to images
import glob
from os import path

# For data augmentation
from random import choice
from torchvision import transforms
import torchvision.transforms.functional as TF

# To create the class
from torch.utils.data.dataset import Dataset

# Utilities
from utils import apply_pca

# Others
import numpy as np


class BreastCancerDataset(Dataset):
    def __init__(self, data_dir, transforms = transforms.ToTensor(), angles = None, pca = None):
        """
        Dataset created from images where the class is embedded in the file name
        Args:
            data_dir: Path to the images
            transform: List of transformations to apply to the images
        """
        if not path.isdir(data_dir):
            raise OSError ('Directory not found')

        # Image list
        self.image_list = glob.glob(data_dir + '*')

        # Number of images
        self.data_len = len(self.image_list)
            
        # Function to transform images
        self.transform = transforms

        # List containing a discrete rotation set
        self.angles = angles

        # PCA matrix (3 channels)
        self.pca = pca

    def __getitem__(self,index):
        
        # Get image name from list
        img_path = self.image_list[index]

        if self.pca is not None:
            
            # Load image using OpenCV (not PIL, this is done this way to use PCA correctly)
            img = cv2.cvtColor( cv2.imread(img_path), cv2.COLOR_BGR2RGB )

            # Split image into RGB channels
            blue, green, red = cv2.split(img)

            # Normalize
            blue, green, red = blue/255, green/255, red/255
            
            # Apply PCA
            # Note: This returns a tensor, and cannot be passed to the ToTensor() transform
            img = apply_pca(red, green, blue, self.pca)

        else:

            # Open image with PIL
            img = Image.open(img_path)

        # Apply transforms 
        tensor = self.transform(img).float()

        # Rotate the image.
        if self.angles is not None:

            # Get a random angle (equal probabilities for all)
            agl = choice(self.angles)

            # Rotate
            tensor = TF.rotate(tensor,agl)

        # Get image label from the file name
        label = int(img_path [-5])

        return (tensor, label)

    def __len__(self):
        return self.data_len    