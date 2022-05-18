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
    def __init__(self, model: torch.nn.Module):

        self.model = model

        self.transforms =  transforms.Compose([

            # Convert to tensor
            transforms.ToTensor()

        ])

        return
    
    def build_histopathological_image(self, dir: str, dst: str):

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

    def build_truelabel_histopathological_image(self, dir: str, dst: str):

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

    def build_predicted_histopathological_image(self, dir: str, dst: str):
        
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