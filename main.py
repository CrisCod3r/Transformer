# PyTorch
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import  DataLoader

# Dataset
from dataset import BreastCancerDataset

# Training and testing loops
from train import *

# Utils
from utils import interval95, compute_and_plot_stats, build_optimizer, build_model, build_transforms, load_pca_matrix

# Others
import argparse as arg
import os

parser = arg.ArgumentParser(description= 'Train or test a CNN or ViT with the breast cancer dataset.')

# Path to training data
parser.add_argument('-tr', '--training_path', dest = 'training_path', default = None, type=str, help= 'Path to training dataset.')

# Path to test data
parser.add_argument('-te', '--test_path', dest = 'test_path', default = None, type=str, help= 'Path to test dataset (can also be path to validation data).')

# Model
parser.add_argument('-n', '--net', dest = 'net', default = None, type=str, help= 'Model to train')

# Number of epochs
parser.add_argument('-e', '--epochs', dest= 'num_epochs', default= 1, type=int, help= "Number of epochs in training")

# Batch size
parser.add_argument('-b', '--batch_size',dest = 'batch_size', default= 8, type=int, help= 'Batch size')

# Optimizer
parser.add_argument('-o', '--optimizer', dest = 'optimizer', default="adam", type=str, help= 'Learning rate optimizer')

# Resume training from checkpoint
parser.add_argument('-r', '--resume', action= 'store_true', dest = 'resume', default=False, help= 'Resume training from checkpoint')

# Test the neural network (requiers the -n parameter)
parser.add_argument('-t', '--test', action= 'store_true', dest = 'test', default=False, help= 'Test a neural network (requiers the -n and -te parameter)')

# Name used for the files generated as output (plots)
parser.add_argument('-na', '--name', dest = 'file_name', default="output", type=str, help= 'Name used for the files generated as output (plots)')

# Number of components for PCA projection
parser.add_argument('-p', '--pca', dest = 'pca_components', default=None, type=int, help= 'Number of components for PCA projection. If not used, no PCA projection will be applied')


# --------------- Global variables ---------------

model_name = ""

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Criterion
criterion = torch.nn.BCEWithLogitsLoss()

# Scheduler
scheduler = None

best_accuracy = 0.0


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
        args: Arguments passed from the argument parser
    """
    global best_accuracy, best_class_accuracy, file_name, n_components, model_name, model, optimizer, pca, trainloader, testloader, scheduler, test_samples

    # Obtain model name
    model_name = args.net.lower()

    # Obtain output file name
    file_name = args.file_name
    
    # Obtain number of components for PCA dimensionality reduction
    pca_components = args.pca_components

    # Model
    print('Building model...')
    model = build_model(model_name)

    if os.path.isfile('./pretrained/' + file_name + '.pth'):
        print("Previous training with this models found. Obtaining best accuracy...")
        state_dict = torch.load('./pretrained/' + file_name + '.pth')
        best_accuracy = state_dict['accuracy']
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
    optimizer = build_optimizer(model, optimizer) 

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
    training_data = BreastCancerDataset(args.training_path + '/', transfs = train_transform, angles = list(range(-90,91,15)), pca = pca)
    print("Loaded %d images" % len(training_data))

    # Get test dataset (13600 images)
    print("Loading test dataset...")
    test_data = BreastCancerDataset(args.test_path + '/', transfs = test_transform, pca = pca)
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
    test_data = BreastCancerDataset(args.test_path, transfs = test_transform, pca = pca)
    print("Loaded %d images" % len(test_data))
    test_samples = len(test_data)


    # Test data loader
    testloader = DataLoader(dataset = test_data, batch_size = args.batch_size, shuffle = False, transform = test_transform)

    return 



def train_model(num_epochs: int) -> None:
    """
    Trains the neural network for a number of epochs.
    Args:
        num_epochs (int): Number of epochs

    Returns: None
    """
    global best_accuracy, model_name, model, trainloader, testloader, optimizer, scheduler, test_samples, file_name
    

    
    for epoch in range(num_epochs):

        train(criterion, device, epoch, model, model_name, optimizer, scheduler, trainloader)

        test_acc = test(best_accuracy, criterion, device, epoch, file_name, model, model_name, testloader)

        if test_acc > best_accuracy:
            best_accuracy = test_acc 


    # Get model when it had the best accuracy
    del model
    model = build_model(model_name)
    model.load_state_dict( torch.load('./pretrained/' + file_name + '.pth')['model'] )
    model.to(device)

    # Obtain predictions
    print("Obtaining predictions...")
    true_labels, predicted_labels, probabilities = predict(device, model, model_name, testloader)

    # Compute and plot metrics
    precision, recall, specificity, f_score, bac = compute_and_plot_stats(true_labels, predicted_labels, probabilities, file_name)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Specificity:", specificity)
    print("F - Score:", f_score)
    print("Balanced Accuracy:", bac)

    # Confidence interval
    interval = interval95( best_accuracy / 100, test_samples)
    print("Confidence interval (95%):")
    print(str(best_accuracy) + ' +- ' + str(interval * 100))



def test_model():

    print("Obtaining predictions...")

    # Obtain predictions
    true_labels, predicted_labels, probabilities = predict(device, model, model_name, testloader)

    # Compute and plot metrics
    precision, recall, specificity, f_score, bac = compute_and_plot_stats(true_labels, predicted_labels, probabilities, file_name)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Specificity:", specificity)
    print("F - Score:", f_score)
    print("Balanced Accuracy:", bac)

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