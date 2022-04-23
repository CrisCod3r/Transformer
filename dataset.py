from PIL import Image 
import glob
from os import path
from random import choice

from torchvision import transforms
from torch.utils.data.dataset import Dataset

import torchvision.transforms.functional as TF


class BreastCancerDataset(Dataset):
    def __init__(self, data_dir, transforms = transforms.ToTensor(), angles = None):
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

    def __getitem__(self,index):
        
        # Get image name from list
        img_path = self.image_list[index]

        # Open image
        img = Image.open(img_path)

        # This implements a discrete rotation to the image.
        if self.angles is not None:
            agl = choice(self.angles)
            img = TF.rotate(img,agl)

        # Apply transforms 
        tensor = self.transform(img).float()

        # Get image label from the file name
        label = int(img_path [-5])

        return (tensor, label)

    def __len__(self):
        return self.data_len

    def apply_pca(self):
        pass
        
    def apply_tsne(self):
        pass

    