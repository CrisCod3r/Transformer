import torch

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms


from dataset import BreastCancerDataset

from models.AlexNet import *
from models.DenseNet import *
from models.DLA import *
from models.EfficientNet import *
from models.GoogLeNet import *
from models.LeNet5 import *
from models.ResNet import *
from models.VGG import *
from models.WeightedNet import WeightedNet

from train import *

from utils import interval95, plot, get_mean_and_std, compute_and_plot_stats

from torch import optim, device, Generator
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, ExponentialLR
from torch.utils.data import Dataset, DataLoader, random_split

import argparse as arg
import sys
import os

parser = arg.ArgumentParser(description= 'Train a CNN with the breast cancer dataset.')

# Path to data
parser.add_argument('-d', '--data', dest = 'path', default = 'data', type=str, help= 'Path to dataset')

# Neural network
parser.add_argument('-n', '--net', dest = 'net', default = 'lenet', type=str, help= 'Neural network to train')

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



# --------------- Global variables ---------------

net_models = {'alexnet':AlexNet(),'densenet':models.densenet121(),'dla': DLA(),'efficientnet':EfficientNet("b7", num_classes=2),
'inception':GoogLeNet(),'lenet':LeNet5(),'resnet':ResNet50(),'vgg':VGG(vgg_type="VGG19"), 'weightednet': WeightedNet()}

optimizers = {"sgd": torch.optim.SGD,"adam":torch.optim.Adam, "adadelta": torch.optim.Adadelta, "adagrad": torch.optim.Adagrad}

model_name = ""

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Criterion
criterion = torch.nn.CrossEntropyLoss()

# Scheduler
scheduler = None

best_accuracy = 0
trainloader, testloader = None, None
model, optimizer = None,None

# ---------------------------------------------
def set_up_training(args):
    """
    Sets up all of the necessary variable for training
    Args:
        device: Device used for traning ('cuda' or 'cpu')
        args: Arguments passed from the argument parser
    """
    global best_accuracy, model_name, model, optimizer, trainloader, testloader, scheduler
    model_name = args.net.lower()




    try:
        # Model
        print('Building model..')
        model = net_models[model_name]

        if os.path.isfile('./pretrained/' + model_name + '.pth'):
            print("Previous training with this models found. Obtaining best accuracy...")
            state_dict = torch.load('./pretrained/' + model_name + '.pth')
            best_accuracy = state_dict['accuracy']
            print("Best accuracy: ", best_accuracy)

        if args.resume:
            state_dict = torch.load('./pretrained/' + model_name + '.pth')
            model.load_state_dict( state_dict['model'] )
            best_accuracy = state_dict['accuracy']
            print("Loaded checkpoint, best accuracy obtained previously is: %.3f" % best_accuracy)

        model.to(device)

    except:
        print("Error, unrecognized model")
        print("Available models:", ', '.join(['alexnet','densenet','dla', 'efficientnet', 'inception', 'lenet', 'resnet', 'vgg','weightednet']))
        sys.exit(-1)

    optimizer = args.optimizer.lower()

    try:
        # Build optimizer
        optimizer = optimizers[optimizer](model.parameters(), lr= 1e-3)

    except:
        print("Error, unrecognized optimizer")
        print("Available optimizers:", ', '.join(['sgd','adam','adadelta','adagrad']))
        sys.exit(-1)

    # Update scheduler
    # scheduler = StepLR(optimizer, step_size = 5, gamma = 0.5)
    # scheduler = ReduceLROnPlateau(optimizer, mode = 'max', factor = 0.5, patience = 5, verbose=True)
    scheduler = ExponentialLR(optimizer, gamma = 0.99,verbose=True)

    # Train transformations
    train_transform = transforms.Compose([

        # Allow random image flips
        # transforms.RandomRotation(degrees = (-90, 90) ),

        # Allow random horizontal flips (data augmentation)
        transforms.RandomHorizontalFlip(p = 0.25),

        # Allow random vertical flips (data augmentation)
        transforms.RandomVerticalFlip(p = 0.05),

        # Transform images to tensors
        transforms.ToTensor(),

        # Normalize train dataset with its mean and standard deviation
        transforms.Normalize((0.7595, 0.5646, 0.6882), (0.1496, 0.1970, 0.1428))

        
    ])

    test_transform = transforms.Compose([
        # Transform images to tensors
        transforms.ToTensor(),

        # Normalize test dataset with its mean and standard deviation
        transforms.Normalize((0.7594, 0.5650, 0.6884), (0.1504, 0.1976, 0.1431))
    ])
    
    # Get training dataset (122400 images)
    print("Loading training dataset...")
    training_data = BreastCancerDataset(args.path + 'train/', transforms = train_transform, angles = [0, 90, -90])
    print("Loaded %d images" % len(training_data))

    # Get validation dataset (13600 images)
    print("Loading validation dataset...")
    test_data = BreastCancerDataset(args.path + 'validation/', transforms = test_transform)
    print("Loaded %d images" % len(test_data))


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
    global model_name, model, testloader
    model_name = args.net.lower()
    
    try:
        # Model
        print('Loading model..')
        model = net_models[model_name]
        model.load_state_dict( torch.load('./pretrained/' + model.name + '.pth')['model'] )
        model.to(device)

    except KeyError:
        print("Error, unrecognized model")
        print("Available models:", ', '.join(['alexnet', 'densenet','dla','efficientnet', 'inception', 'lenet', 'resnet', 'vgg','weightednet']))
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

    return 



def train_model(num_epochs):
    """
    Trains the neural network
    Args:
        num_epochs: Number of epochs
    """
    global best_accuracy, model_name, model, trainloader, testloader, optimizer, scheduler
    
    train_loss_list, test_loss_list = [], []
    train_acc_list, test_acc_list = [], []
    best_class_accuracy = []

    # Weights in case the model is WeightedNet
    if model.name == "WeightedNet":
        weights_list = [model.weights()]

    classes = ('benign','malignant')
        
    for epoch in range(num_epochs):

        train_loss, train_acc = train(criterion, device, epoch, model, optimizer, scheduler, trainloader)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        if model.name == "WeightedNet":
            weights_list.append( model.weights())

        test_loss, test_acc, class_accuracy = test(best_accuracy, classes, criterion, device, epoch, model, optimizer, testloader)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)   

        if test_acc > best_accuracy:
            best_accuracy = test_acc 
            best_class_accuracy = class_accuracy   

    if model.name == "WeightedNet":
        weights = []
        weights_list = list(map(list,zip(*weights_list)))

        for i in range(0,len(weights_list)):
            weights.append( ( weights_list[i], "Clasificador " + str(i+1) ) )

        plot( list(range(0, num_epochs + 1)),weights, "Epochs", "Weights", name= "Weights_" + model.name)


    

    # Plot confussion matrix when the model had the best accuracy
    del model
    model = net_models[model_name]
    model.load_state_dict( torch.load('./pretrained/' + model.name + '.pth')['model'] )
    model.to(device)

    print("Obtaining predictions...")
    true_labels, predicted_labels = test_and_return(device, model, testloader)

    precision, recall, specificity, f_score, bac = compute_and_plot_stats(true_labels, predicted_labels, model_name)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Specificity:", specificity)
    print("F - Score:", f_score)
    print("Balanced Accuracy:", bac)
    print("Best accuracy: ", best_accuracy)

    interval = interval95( best_accuracy / 100, len(testloader))
    print("Confidence interval (95%):")
    print(str(best_accuracy) + ' +- ' + str(interval * 100))

    for idx in range(len(classes)):
        print("Accuracy of class " + classes[idx] + ": %.3f" % best_class_accuracy[idx])

def test_model():

    print("Calculating accuracy...")
    
    classes = ('benign','malignant')
    test_acc, class_accuracy = final_test(classes, device, model, testloader)


    interval = interval95( best_accuracy / 100, len(testloader))

    print("Accuracy: ", test_acc)
    print("Confidence interval (95%):")
    print("[%.3f, %.3f]" % (best_accuracy - interval[0], best_accuracy + interval[1]) )

    for idx in range(len(classes)):
        print("Accuracy of class " + classes[idx] + ": %.3f" % class_accuracy[idx])




def main():



    args = parser.parse_args()

    if not args.test:
        set_up_training(args)
        train_model(args.num_epochs)
    else:
        setup_test(args)
        test_model()



    

if __name__ == "__main__":
    main()