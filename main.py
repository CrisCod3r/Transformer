import torch
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms


from dataset import BreastCancerDataset

from models.AlexNet import *
from models.EfficientNet import *
from models.GoogLeNet import *
from models.LeNet5 import *
from models.ResNet import *
from models.VGG import *
from models.WeightedNet import WeightedNet
from train import *

from utils import interval95, plot, get_mean_and_std

from torch import optim, device, Generator
from torch.utils.data import Dataset, DataLoader, random_split

import argparse as arg
import sys
import os

parser = arg.ArgumentParser(description= 'Train a CNN with the breast cancer dataset.')

# Path to data
parser.add_argument('-d', '--data', dest = 'path', default = 'data', type=str, help= 'Path to dataset')

# Neural network
parser.add_argument('-n', '--net', dest = 'net', default = 'weightednet', type=str, help= 'Neural network to train')

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


def set_up_training(device, args):
    """
    Sets up all of the necessary variable for training
    Args:
        device: Device used for traning ('cuda' or 'cpu')
        args: Arguments passed from the argument parser
    """

    model = args.net.lower()

    net_models = {'alexnet':AlexNet(),'efficientnet':EfficientNet("b7", num_classes=2),
    'inception':GoogLeNet(),'lenet':LeNet5(),'resnet':ResNet50(),'vgg':VGG(vgg_type="VGG19"), 'weightednet': WeightedNet()}

    try:
        # Model
        print('Building model..')
        model = net_models[model]
        best_accuracy = 0

        if args.resume:
            state_dict = torch.load('./pretrained/' + model.name + '.pth')
            model.load_state_dict( state_dict['model'] )
            best_accuracy = state_dict['accuracy']
            print("Loaded checkpoint, best accuracy obtained previously is: %.3f" % best_accuracy)

        model.to(device)

    except:
        print("Error, unrecognized model")
        print("Available models:", ', '.join(['alexnet', 'efficientnet', 'inception', 'lenet', 'resnet', 'vgg','weightednet']))
        sys.exit(-1)

    optimizers = {"sgd": torch.optim.SGD,"adam":torch.optim.Adam, "adadelta": torch.optim.Adadelta, "adagrad": torch.optim.Adagrad}
    optimizer = args.optimizer.lower()

    try:
        # Build optimizer
        optimizer = optimizers[optimizer](model.parameters(), lr= 1e-3)

    except:
        print("Error, unrecognized optimizer")
        print("Available optimizers:", ', '.join(['sgd','adam','adadelta','adagrad']))
        sys.exit(-1)

    
    # Criterion
    criterion = torch.nn.CrossEntropyLoss()

    # Train transformations
    train_transform = transforms.Compose([

        # Transform images to tensors
        transforms.ToTensor(),

        # Normalize train dataset with its mean and standard deviation
        transforms.Normalize((0.7595, 0.5646, 0.6882), (0.1496, 0.1970, 0.1428)),

        # Allow random flips (data augmentation)
        transforms.RandomHorizontalFlip()
    ])

    test_transform = transforms.Compose([
        # Transform images to tensors
        transforms.ToTensor(),

        # Normalize test dataset with its mean and standard deviation
        transforms.Normalize((0.7594, 0.5650, 0.6884), (0.1504, 0.1976, 0.1431))
    ])
    
    # Get training dataset (122400 images)
    print("Loading training dataset...")
    training_data = BreastCancerDataset(args.path + 'train/', transforms = train_transform)
    print("Loaded %d images" % len(training_data))

    # Get validation dataset (13600 images)
    print("Loading validation dataset...")
    test_data = BreastCancerDataset(args.path + 'validation/', transforms = test_transform)
    print("Loaded %d images" % len(test_data))


    # Training data loader
    trainloader = DataLoader(dataset = training_data, batch_size = args.batch_size, shuffle = True)

    # Test data loader
    testloader = DataLoader(dataset = test_data, batch_size = args.batch_size, shuffle = False)

    return best_accuracy, criterion, model, optimizer, trainloader, testloader




def setup_test(device,args):
    """
    Sets up all of the necessary variable for testing
    Args:
        device: Device used for traning ('cuda' or 'cpu')
        args: Arguments passed from the argument parser
    """
    model = args.net.lower()
    
    net_models = {'alexnet':AlexNet(),'efficientnet':EfficientNet("b7", num_classes=2),
    'inception':GoogLeNet(),'lenet':LeNet5(),'resnet':ResNet50(),'vgg':VGG(vgg_type="VGG19"), 'weightednet': WeightedNet()}

    try:
        # Model
        print('Loading model..')
        model = net_models[model]
        model.load_state_dict( torch.load('./pretrained/' + model.name + '.pth')['model'] )
        model.to(device)

    except KeyError:
        print("Error, unrecognized model")
        print("Available models:", ', '.join(['alexnet', 'efficientnet', 'inception', 'lenet', 'resnet', 'vgg','weightednet']))
        sys.exit(-1)
    except Exception as e:
        print(e)
        sys.exit(-1)


    test_transform = transforms.Compose([
        # Transform images to tensors
        transforms.ToTensor(),

        # Normalize test dataset with its mean and standard deviation
        transforms.Normalize((0.7585, 0.5622, 0.6866), (0.1492, 0.1961, 0.1423)),

    ])

    # Get test dataset (15110 images)
    print("Loading test dataset...")
    test_data = BreastCancerDataset(args.path, transforms = test_transform)
    print("Loaded %d images" % len(test_data))



    # Test data loader
    testloader = DataLoader(dataset = test_data, batch_size = args.batch_size, shuffle = False, transform = test_transform)

    return model, testloader



def train_model(best_accuracy, criterion, device, model, optimizer, trainloader, testloader, num_epochs):
    """
    Trains the neural network
    Args:
        best_accuracy: Best accuracy obtained (0 if checkpoint has not been loaded)
        criterion: Loss criterion
        device: Device used ('cuda' or 'cpu')
        model: Neural Network
        optimizer: Learning rate optimizer
        trainloader: Training data loader
        testloaer: Test data loader
        num_epochs: Number of epochs
    """

    train_loss_list, test_loss_list = [], []
    train_acc_list, test_acc_list = [], []
    classes = ('benign','malignant')

        
    for epoch in range(num_epochs):

        train_loss, train_acc = train(criterion, device, epoch, model, optimizer, trainloader)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        test_loss, test_acc = test(best_accuracy, classes, criterion, device, epoch, model, optimizer, testloader)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)   

        if test_acc > best_accuracy:
            best_accuracy = test_acc    

    # If couldn't plot correctly
    if not plot(list(range(1, num_epochs + 1)), [ (train_loss_list, 'Train loss'), (test_loss_list, 'Test loss') ], "Epochs", "Loss", name = "Loss_"+ model.name):

        print("Train loss:\n", ','.join(train_loss_list))
        print("Test loss:\n", ','.join(test_loss_list))

    if not plot(list(range(1, num_epochs + 1)),[ (train_acc_list, 'Train accuracy'), (test_acc_list, 'Test accuracy') ] , "Epochs", "Accuracy (%)", name = "Accuracy_"+ model.name):
        print("Train accuracy:\n", ','.join(train_acc_list))
        print("Test accuracy:\n", ','.join(test_acc_list))
    
    print("Best accuracy: ", best_accuracy)

    interval = interval95( best_accuracy / 100, len(test_data))
    print("Confidence interval (95%):")
    print("[%.3f, %.3f]" % best_accuracy - interval[0], best_accuracy + interval[1] )

def test_model(device, model, testloader):

    print("Calculating accuracy...")
    
    classes = ('benign','malignant')
    test_acc, class_accuracy = final_test(classes, device, model, testloader)


    interval = interval95( best_accuracy / 100, len(test_data))

    print("Accuracy: ", test_acc)
    print("Confidence interval (95%):")
    print("[%.3f, %.3f]" % best_accuracy - interval[0], best_accuracy + interval[1] )

    for idx in range(len(classes)):
        print("Accuracy of class " + classes[idx] + ": %.3f" % class_accuracy[idx])




def main():

    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args = parser.parse_args()

    if not args.test:
        best_accuracy, criterion, model, optimizer, trainloader, testloader = set_up_training(device,args)
        train_model(best_accuracy, criterion, device, model, optimizer, trainloader, testloader, args.num_epochs)
    else:
        model, testloader = setup_test(device,args)
        test_model(device, model, testloader)



    

if __name__ == "__main__":
    main()