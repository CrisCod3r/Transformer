from PIL import Image 
import glob
from os import path

from torchvision import transforms
from torch.utils.data.dataset import Dataset


class BreastCancerDataset(Dataset):
    def __init__(self, data_dir, transforms = transforms.ToTensor()):
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

    def __getitem__(self,index):
        
        # Get image name from list
        img_path = self.image_list[index]

        # Open image
        img = Image.open(img_path)

        # Apply transforms 
        tensor = self.transform(img).float()

        # Get image label from the file name
        label = int(img_path [-5])

        return (tensor, label)

        
        

    def __len__(self):
        return self.data_len

    