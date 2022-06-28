#PyTorch
import torch
import torchvision.models as torch_models

# HistopathologyImageMaker
from HistImageMaker import HistopathologyImageMaker as histmaker

# Others
import os
import argparse as arg

parser = arg.ArgumentParser(description= 'Build original, true and predicted histopathological images.')

# Path to images
parser.add_argument('-p', '--path', dest = 'path', default = None, type=str, help= 'Path to images.')

# Path to images
parser.add_argument('-d', '--destination', dest = 'dest', default = None, type=str, help= 'Destination directory.')

def main():

    # Parse arguments
    args = parser.parse_args()
    args.path += '/'
    
    # If directory doesn't exist, make one
    if not os.path.isdir(args.dest): os.mkdir(args.dest)

    # Best accuracy was obtanied by EfficientNetB6
    model = torch_models.efficientnet_b6()

    if torch.cuda.is_available():
        model.load_state_dict(torch.load('./pretrained/EfficientNetB6.pth') ['model'])
    else:
        model.load_state_dict(torch.load('./pretrained/EfficientNetB6.pth', map_location=torch.device('cpu')) ['model'])

    # Make HistopathologyImageMaker
    histimgmaker = histmaker(model)

    for dir in os.listdir(args.path):

        # Build original image
        histimgmaker.build_histopathological_image(args.path + dir + '/', args.dest + '/' + dir + '_original')

        # Build image with true labels (only malignant)
        histimgmaker.build_truelabel_histopathological_image(args.path + dir + '/', args.dest + '/' + dir + '_true')

        # Build image with predicted labels (only malignant)
        histimgmaker.build_predicted_histopathological_image(args.path + dir + '/', args.dest + '/' + dir + '_predicted')

if __name__ == "__main__":
    main()