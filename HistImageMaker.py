# PyTorch
import torch
import torchvision.transforms as transforms

# Image managing
from PIL import Image

# Paths
import os

# Utils
from tqdm import tqdm

class HistopathologyImageMaker:
    def __init__(self, model: torch.nn.Module) -> None:
        """
        Constructor for the HistopathologyImageMaker

        Args:
            model (torch.nn.Module): Model that will be used to predict the patches in the reconstruction.

        Raises:
            TypeError: The given model is not a torch.nn.Module
        """  

        if not isinstance(model, torch.nn.Module): raise TypeError('"model" must be a torch.nn.Module')

        self.model = model

        self.transforms =  transforms.Compose([

            # Convert to tensor
            transforms.ToTensor(),

            # Normalize train dataset with its mean and standard deviation
            transforms.Normalize((0.7595, 0.5646, 0.6882), (0.1496, 0.1970, 0.1428))
        ])

        return
    
    def build_histopathological_image(self, dir: str, dst: str) -> None:
        """
        Builds an histopathological image given its patches

        Args:
            dir (str): Path to the image patches
            dst (str): Name of destination image

        Raises:
            TypeError: The given path is not a string
            TypeError: The given destination is not a string
            OSError: The given path is not a directory

        Returns:
            None
        """        
        if not os.path.isdir(dir): raise OSError ('"dir" is not a directory')
        if not isinstance(dir, str): raise TypeError('"dir" must be a string')
        if not isinstance(dst, str): raise TypeError('"dst" must be a string')

        # Get image paths
        img_dir = os.listdir(dir)

        x_list, y_list = [], []
        max_x, max_y = 0, 0
        for img in tqdm(img_dir, desc = "Gathering data..."):
            
            x, y = img[1:-10].split('y')
            x,y = int(x [:-1]), int(y [:-1])

            x_list.append(x)
            y_list.append(y)

            if x > max_x:
                max_x = x
                
            if y > max_y:
                max_y = y

        hist_image = Image.new(mode = 'RGB', size =  (max_x, max_y), color = (255,255,255))

        for idx in tqdm(range(len(img_dir)), desc = "Building image..."):
        

            img = Image.open(dir + img_dir[idx])
            hist_image.paste(img, (x_list[idx], y_list[idx]))

        hist_image.save(dst + ".png")
        return 

    def build_truelabel_histopathological_image(self, dir: str, dst: str) -> None:
        """
        Builds an histopathological image given its patches and its labels

        Args:
            dir (str): Path to the image patches
            dst (str): Name of destination image

        Raises:
            TypeError: The given path is not a string
            TypeError: The given destination is not a string
            OSError: The given path is not a directory
        
        Returns:
            None
        """        

        if not os.path.isdir(dir): raise OSError ('"dir" is not a directory')
        if not isinstance(dir, str): raise TypeError('"dir" must be a string')
        if not isinstance(dst, str): raise TypeError('"dst" must be a string')

        # Get image paths
        img_dir = os.listdir(dir)

        x_list, y_list = [], []
        max_x, max_y = 0, 0
        for img in tqdm(img_dir, desc = "Gathering data..."):
            
            x, y = img[1:-10].split('y')
            x,y = int(x [:-1]), int(y [:-1])

            x_list.append(x)
            y_list.append(y)

            if x > max_x:
                max_x = x
                
            if y > max_y:
                max_y = y

        hist_image = Image.new(mode = 'RGB', size =  (max_x, max_y), color = (255,255,255))
        benign_patch = Image.new(mode = 'RGB', size =  (50, 50), color = (255,0,0))
        malignant_patch = Image.new(mode = 'RGB', size =  (50, 50), color = (0,255,0))

        for idx in tqdm(range(len(img_dir)), desc = "Building image..."):
        

            if img_dir[idx] [-5] == "0":
                hist_image.paste(benign_patch, (x_list[idx], y_list[idx]))

            else:
                hist_image.paste(malignant_patch, (x_list[idx], y_list[idx]))

        hist_image.save(dst + ".png")
        return 

    def build_predicted_histopathological_image(self, dir: str, dst: str) -> None:
        """
        Builds an histopathological image given its patches using the model to predict the label

        Args:
            dir (str): Path to the image patches
            dst (str): Name of destination image

        Raises:
            TypeError: The given path is not a string
            TypeError: The given destination is not a string
            OSError: The given path is not a directory

        Returns:
            None
        """        
        if not os.path.isdir(dir): raise OSError ('"dir" is not a directory')
        if not isinstance(dir, str): raise TypeError('"dir" must be a string')
        if not isinstance(dst, str): raise TypeError('"dst" must be a string')

        # Get image paths
        img_dir = os.listdir(dir)

        x_list, y_list = [], []
        max_x, max_y = 0, 0
        for img in tqdm(img_dir, desc = "Gathering data..."):
            
            x, y = img[1:-10].split('y')
            x,y = int(x [:-1]), int(y [:-1])

            x_list.append(x)
            y_list.append(y)

            if x > max_x:
                max_x = x
                
            if y > max_y:
                max_y = y

        hist_image = Image.new(mode = 'RGB', size =  (max_x, max_y), color = (255,255,255))
        benign_patch = Image.new(mode = 'RGB', size =  (50, 50), color = (255,0,0))
        malignant_patch = Image.new(mode = 'RGB', size =  (50, 50), color = (0,255,0))

        self.model.eval()
        with torch.no_grad():
            for idx in tqdm(range(len(img_dir)), desc = "Building image..."):
                
                # Open image
                img = Image.open(dir + img_dir[idx])

                # Transform to tensor
                tensor = self.transforms(img)

                # Add an extra dimension to allow forwarding through the model
                tensor = tensor[None]

                # Predict label
                output = self.model(tensor)
                _, predicted = output.max(1)
                predicted = predicted.item()

                # Prediction == Benign
                if predicted == 0:
                    hist_image.paste(benign_patch, (x_list[idx], y_list[idx]))

                # Prediction == Malignant
                else:
                    hist_image.paste(malignant_patch, (x_list[idx], y_list[idx]))

        # Save image
        hist_image.save(dst + ".png")

        return 