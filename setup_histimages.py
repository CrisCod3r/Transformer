# Gathers all images and separates then by folders
# The zip file must be named "archive.zip" and must be in the same directory

# To unzip the file
import zipfile as zf

# File managing
import shutil
import os

# Utils
from tqdm import tqdm
from PIL import Image




# Create the necesary folders.
# If "archive" directory already exists, it is overwritten
data_folder = "./archive/"
os.makedirs(data_folder,exist_ok=True)
os.chdir(data_folder)
print("Created data folder")
print("Preparing to extract...")

# Extract files from zip file
with zf.ZipFile('../archive.zip','r') as arc:
    for member in tqdm(arc.infolist(),desc='Extracting...'):
        try:
            arc.extract(member,'.')
        except:
            pass


os.chdir("..")

# There is a folder named 'IDC_regular_ps50_idx5' in the zip file
# However it is not necesary, so we just remove it
for i in tqdm(os.listdir(data_folder + 'IDC_regular_ps50_idx5/'),desc = 'Removing unnecesary files...'):
    shutil.rmtree(data_folder + 'IDC_regular_ps50_idx5/' + i)

# Once files have been removed, we delete the directory   
os.rmdir(data_folder + 'IDC_regular_ps50_idx5/')

dirs = os.listdir(data_folder)   

img_idx = 1
for fold in tqdm(dirs, desc= 'Renaming and moving files...'):

    class_0_folder = data_folder + str(fold) + "/0/"
    class_1_folder = data_folder + str(fold) + "/1/"

   # Class 0 samples
    for img_name in os.listdir(class_0_folder):

        # Open image
        img = Image.open(class_0_folder + img_name)

        # Check size
        if img.size [0] == 50 and img.size [1] == 50:

            # Form new name
            aux = img_name.split("_")
            new_name = '_'.join(aux[2:])

            # Rename file and move
            os.rename(class_0_folder + img_name, class_0_folder + new_name )
            shutil.move(class_0_folder + new_name, data_folder + str(fold) + '/' + new_name)

        else:
            # If file is corrupted, remove
            os.remove(class_0_folder + img_name)

    # Class 1 samples
    for img_name in os.listdir(class_1_folder):

        # Open image
        img = Image.open(class_1_folder + img_name)

        # Check size
        if img.size [0] == 50 and img.size [1] == 50:

            # Form new name
            aux = img_name.split("_")
            new_name = '_'.join(aux[2:])

            # Rename file and move
            os.rename(class_1_folder + img_name, class_1_folder + new_name)
            shutil.move(class_1_folder + new_name, data_folder + str(fold) + '/' + new_name)

        else:
            # If file is corrupted, remove
            os.remove(class_1_folder + img_name)

    # Remove the "0" and "1" folders
    shutil.rmtree(class_0_folder)
    shutil.rmtree(class_1_folder)

    # Renae folder and update index
    os.rename(data_folder + str(fold), data_folder + str(img_idx))
    img_idx += 1
    

print("Setup succesful")