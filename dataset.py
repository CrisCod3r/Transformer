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
from typing import Tuple
from torch import Tensor

class BreastCancerDataset(Dataset):
    def __init__(self, data_dir: str, transfs: transforms.transforms.Compose = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.7595, 0.5646, 0.6882), (0.1497, 0.1970, 0.1428)) ]), 
                angles: list = None, pca: dict = None):
        """
        Dataset for breast histopathology images where the label is embedded in the file.

        Args:
            data_dir (str): Path to the images
            transfs (transforms.transforms.Compose, optional): Transform to apply to the images for data augmentation. Defaults to transforms.ToTensor().
            angles (list, optional): List of integers which each one indicates an angle to perform a discrete rotation. None means no rotations will be done. Defaults to None.
            pca (dict, optional): PCA matrix for all 3 color channels. None means no PCA will be applied.Defaults to None.

        Raises:
            OSError: Path to images not found
            TypeError: The given path is not a string
            TypeError: The given transforms are not torchvision.transforms.transforms.Compose.
            TypeError: The given angles are not a list
            TypeError: The given PCA matrix is not a dictionary.
        """        
        if not path.isdir(data_dir):
            raise OSError ('Directory not found')
        
        if not isinstance(data_dir, str): raise TypeError('"data_dir" must be a str.')
        if not isinstance(transfs, transforms.transforms.Compose): raise TypeError('"transfs" must be a torchvision.transforms.transforms.Compose.')
        if not ( angles is None or isinstance(angles, list) ): raise TypeError('"angles" must be a list of integer or None.')
        if not ( pca is None or isinstance(pca, dict) ): raise TypeError('"pca" must be a dictionary or None.')

        # Image list
        self.image_list = glob.glob(data_dir + '*')

        # Number of images
        self.data_len = len(self.image_list)
            
        # Function to transform images
        self.transfs = transfs

        # List containing a discrete rotation set
        self.angles = angles

        # PCA matrix (3 channels)
        self.pca = pca

    def __getitem__(self,index: int) -> Tuple[Tensor, int]:
        """
        Returns an image given a index

        Args:
            index (int): The index of the image

        Returns:
            Tensor: The image as a Tensor
            int: The label of the image
        """        
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
        tensor = self.transfs(img).float()

        # Rotate the image.
        if self.angles is not None:

            # Get a random angle (equal probabilities for all)
            agl = choice(self.angles)

            # Rotate
            tensor = TF.rotate(tensor,agl)

        # Get image label from the file name
        label = int(img_path [-5])

        return (tensor, label)

    def __len__(self) -> int:
        """
        Returns the number of images in the dataset

        Returns:
            int: The length of the dataset
        """        
        return self.data_len    