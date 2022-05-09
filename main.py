import torch

import torchvision
import torchvision.transforms as transforms


from dataset import BreastCancerDataset

from train import *

from utils import interval95, plot, get_mean_and_std, compute_and_plot_stats, build_optimizer, build_model, build_transforms, load_pca_matrix

from torch import optim, device, Generator
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, ExponentialLR
from torch.utils.data import Dataset, DataLoader, random_split

import argparse as arg
import os

parser = arg.ArgumentParser(description= 'Train or test a CNN or ViT with the breast cancer dataset.')

# Path to data
parser.add_argument('-d', '--data', dest = 'path', default = 'data', type=str, help= 'Path to dataset')

# Model
parser.add_argument('-n', '--net', dest = 'net', default = 'lenet', type=str, help= 'Model to train')

# Number of epochs
parser.add_argument('-e', '--epochs', dest= 'num_epochs', default= 1, type=int, help= "Number of epochs in training")

# Batch size
parser.add_argument('-b', '--batch_size',dest = 'batch_size', default= 8, type=int, help= 'Batch size')

# Optimizer
parser.add_argument('-o', '--optimizer', dest = 'optimizer',default="adam", type=str,help= 'Learning rate optimizer')

# Resume training from checkpoint
parser.add_argument('-r', '--resume', action= 'store_true',dest = 'resume',default=False,help= 'Resume training from checkpoint')

# Test the neural network (requiers the -n parameter)
parser.add_argument('-t', '--test', action= 'store_true',dest = 'test',default=False,help= 'Test the neural network (requiers the -n parameter)')

# Name used for the files generated as output (plots)
parser.add_argument('-na', '--name', dest = 'file_name',default="output", type=str,help= 'Name used for the files generated as output (plots)')


# Mutually exclusive group for PCA and t-SNE
projection_group = parser.add_mutually_exclusive_group()

# Number of components for PCA projection
projection_group.add_argument('-p', '--pca', dest = 'pca_components',default=None, type=int,help= 'Number of components for PCA projection. Can not be used with t-SNE')

# Number of components for t-SNE projection
projection_group.add_argument('-s', '--sne', dest = 'sne_components',default=None, type=int,help= 'Number of components for t-SNE projection. Can not be used with PCA.')

# --------------- Global variables ---------------

model_name = ""

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Criterion
criterion = torch.nn.CrossEntropyLoss()

# Scheduler
scheduler = None

best_accuracy = 0
best_class_accuracy = []


trainloader, testloader = None, None
model, optimizer = None,None

test_samples = 0
file_name = None

# PCA Matrix
pca = None

# ---------------------------------------------
def set_up_training(args):
    """
    Sets up all of the necessary variable for training
    Args:
        device: Device used for traning ('cuda' or 'cpu')
        args: Arguments passed from the argument parser
    """
    global best_accuracy, best_class_accuracy, file_name, n_components, model_name, model, optimizer, pca,  trainloader, testloader, scheduler, test_samples

    # Obtain model name
    model_name = args.net.lower()

    # Obtain output file name
    file_name = args.file_name
    
    # Obtain number of components for PCA dimensionality reduction
    pca_components = args.pca_components

    # Obtain number of components for t-SNE dimensionality reduction
    sne_components = args.sne_components

    # Model
    print('Building model...')
    model = build_model(model_name)

    if os.path.isfile('./pretrained/' + file_name + '.pth'):
        print("Previous training with this models found. Obtaining best accuracy...")
        state_dict = torch.load('./pretrained/' + file_name + '.pth')
        best_accuracy = state_dict['accuracy']
        best_class_accuracy = state_dict['class_accuracy']
        print("Best accuracy: ", best_accuracy)

    if args.resume:
        state_dict = torch.load('./pretrained/' + file_name + '.pth')
        model.load_state_dict( state_dict['model'] )
        best_accuracy = state_dict['accuracy']
        print("Loaded checkpoint, best accuracy obtained previously is: %.3f" % best_accuracy)

    model.to(device)


    # Get optimizer
    print('Loading optimizer...')
    optimizer = args.optimizer.lower()
    optimizer = build_optimizer(optimizer) (model.parameters(), lr= 1e-3)

    # Build scheduler
    scheduler = ExponentialLR(optimizer, gamma = 0.95, verbose=True)

    # Data augmentation
    print("Loading data augmentation transforms...")
    train_transform, test_transform = build_transforms(model_name, pca_components is not None)

    # Load PCA matrix
    if pca_components is not None:
        print("Loading PCA matrix...")
        pca = load_pca_matrix(pca_components)    

    # Get training dataset (122400 images) with rotations
    print("Loading training dataset...")
    training_data = BreastCancerDataset(args.path + 'train/', transfs = train_transform, angles = [0, 90, -90], pca = pca)
    print("Loaded %d images" % len(training_data))

    # Get validation dataset (13600 images)
    print("Loading validation dataset...")
    test_data = BreastCancerDataset(args.path + 'validation/', transfs = test_transform, pca = pca)
    print("Loaded %d images" % len(test_data))

    test_samples = len(test_data)

    # Training data loader
    trainloader = DataLoader(dataset = training_data, batch_size = args.batch_size, shuffle = True)

    # Test data loader
    testloader = DataLoader(dataset = test_data, batch_size = args.batch_size, shuffle = False)

    return 




def setup_test(args):
    """
    Sets up all of the necessary variable for testing
    Args:
        device: Device used for traning ('cuda' or 'cpu')
        args: Arguments passed from the argument parser
    """
    global file_name, model_name, model, pca, testloader, test_samples

    # Obtain model name
    model_name = args.net.lower()

    # Obtain output file name
    file_name = args.file_name
    
    # Obtain number of components for PCA dimensionality reduction
    pca_components = args.pca_components

    # Obtain number of components for t-SNE dimensionality reduction
    sne_components = args.sne_components

    # Load PCA matrix
    if pca_components is not None:
        print("Loading PCA matrix...")
        pca = load_pca_matrix(pca_components)    

    # Model
    print('Building model..')
    model = build_model(model_name)
    model.load_state_dict( torch.load('./pretrained/' + file_name + '.pth')['model'] )
    model.to(device)

    _, test_transform = build_transforms(model_name, pca_components is not None)

    # Get test dataset (15110 images)
    print("Loading test dataset...")
    test_data = BreastCancerDataset(args.path, transfs = test_transform, pca = pca)
    print("Loaded %d images" % len(test_data))
    test_samples = len(test_data)


    # Test data loader
    testloader = DataLoader(dataset = test_data, batch_size = args.batch_size, shuffle = False, transform = test_transform)

    return 



def train_model(num_epochs):
    """
    Trains the neural network
    Args:
        num_epochs: Number of epochs
    """
    global best_accuracy, best_class_accuracy, model_name, model, trainloader, testloader, optimizer, scheduler, test_samples, file_name
    
    classes = ('benign','malignant')
        
    for epoch in range(num_epochs):

        train(criterion, device, epoch, model, optimizer, scheduler, trainloader)

        test_acc, class_accuracy = test(best_accuracy, classes, criterion, device, epoch, file_name, model, testloader)

        if test_acc > best_accuracy:
            best_accuracy = test_acc 
            best_class_accuracy = class_accuracy   

    # Plot confussion matrix when the model had the best accuracy
    del model
    model = build_model(model_name)
    model.load_state_dict( torch.load('./pretrained/' + file_name + '.pth')['model'] )
    model.to(device)

    # Obtain predictions
    print("Obtaining predictions...")
    true_labels, predicted_labels = test_and_return(device, model, testloader)

    # Compute and plot metrics
    precision, recall, specificity, f_score, bac = compute_and_plot_stats(true_labels, predicted_labels, file_name)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Specificity:", specificity)
    print("F - Score:", f_score)
    print("Balanced Accuracy:", bac)
    print("Best accuracy: ", best_accuracy)

    # Confidence interval
    interval = interval95( best_accuracy / 100, test_samples)
    print("Confidence interval (95%):")
    print(str(best_accuracy) + ' +- ' + str(interval * 100))

    for idx in range(len(classes)):
        print("Accuracy of class " + classes[idx] + ": %.3f" % best_class_accuracy[idx])

def test_model():

    print("Obtaining predictions...")

    # Obtain predictions
    true_labels, predicted_labels = test_and_return(device, model, testloader)

    # Compute and plot metrics
    precision, recall, specificity, f_score, bac = compute_and_plot_stats(true_labels, predicted_labels, file_name)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Specificity:", specificity)
    print("F - Score:", f_score)
    print("Balanced Accuracy:", bac)
    print("Best accuracy: ", best_accuracy)

    # Confidence interval
    interval = interval95( best_accuracy / 100, test_samples)
    print("Confidence interval (95%):")
    print(str(best_accuracy) + ' +- ' + str(interval * 100))




def main():

    # Parse arguments
    args = parser.parse_args()

    if not args.test:
        set_up_training(args)
        train_model(args.num_epochs)
    else:
        setup_test(args)
        test_model()



    

if __name__ == "__main__":
    main()