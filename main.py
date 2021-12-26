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
from densenet import *

from train import *

from utils import interval95, plot

from torch import optim, device
from torch.utils.data import Dataset, DataLoader, random_split

import argparse as arg
import sys

parser = arg.ArgumentParser(description= 'Train a CNN with the breast cancer dataset.')

# Path to data
parser.add_argument('-d', '--data', dest = 'path', default = 'data', type=str, help= 'Path to dataset')

# Neural network
parser.add_argument('-n', '--net', dest = 'net', default = 'cnn', type=str, help= 'Neural network to train')

# Number of epochs
parser.add_argument('-e', '--epochs', dest= 'num_epochs', default= 1, type=int, help= "Number of epochs in training")

# Batch size
parser.add_argument('-b', '--batch_size',dest = 'batch_size', default= 8, type=int, help= 'Batch size')

# Optimizer
parser.add_argument('-o', '--optimizer', dest = 'optimizer',default="adam", type=str,help= 'Learning rate optimizer')


def main():

    args = parser.parse_args()

    model = args.net.lower()

    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net_models = {'alexnet':AlexNet(), 'densenet':models.densenet161(),'efficientnet':EfficientNet("b7", num_classes=2),
    'inception':GoogLeNet(),'lenet':LeNet5(),'resnet':ResNet50(),'vgg':VGG(vgg_type="VGG19")}

    try:
        # Model
        print('Building model..')
        model = net_models[model].to(device)

    except:
        print("Error, unrecognized model")
        print("Available models:", ', '.join(['alexnet', 'densenet', 'efficientnet', 'inception', 'lenet', 'resnet', 'vgg']))
        sys.exit(-1)

    optimizers = {"sgd": torch.optim.SGD,"adam":torch.optim.Adam, "adadelta": torch.optim.Adadelta, "adagrad": torch.optim.Adagrad}
    optimizer = args.optimizer.lower()

    try:
        # Build optimizer
        optimizer = optimizers[optimizer](model.parameters(), lr= 1e-3)

    except:
        print("Error, unrecognized optimizer")
        print("Available models:", ', '.join(['sgd','adam','adadelta','adagrad']))
        sys.exit(-1)

    

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    # Criterion
    criterion = torch.nn.CrossEntropyLoss()

    # Get training dataset (136000 images)
    print("Loading datasets...")

    training_data = BreastCancerDataset(args.path)
    training_data, test_data = random_split(training_data, lengths= [122400, 13600])

    trainloader = DataLoader(dataset = training_data, batch_size = args.batch_size, shuffle = True, num_workers=2 if device == 'cuda' else 1)
    testloader = DataLoader(dataset = test_data, batch_size = args.batch_size, shuffle = False, num_workers=2 if device == 'cuda' else 1)

    train_loss_list, test_loss_list = [], []
    train_acc_list, test_acc_list = [], []
    best_accuracy = 0
    classes = ('benign','malignant')

    for epoch in range(args.num_epochs):

        train_loss, train_acc = train(criterion, device, epoch, model, optimizer, trainloader)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        test_loss, test_acc = test(best_acc, classes, criterion, device, epoch, model, optimizer, testloader)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)   

        if test_acc > best_acc:
            best_acc = test_acc    

    # If couldn't plot correctly
    if not plot(list(range(1, args.num_epochs + 1)), [ (train_loss_list, 'Train loss'), (test_loss_list, 'Test loss') ], "Epochs", "Loss", name = "Loss_"+ model.name):

        print("Train loss:\n", ','.join(train_loss_list))
        print("Test loss:\n", ','.join(test_loss_list))

    if not plot(list(range(1, args.num_epochs + 1)),[ (train_acc_list, 'Train accuracy'), (test_acc_list, 'Test accuracy') ] , "Epochs", "Accuracy (%)", name = "Accuracy_"+ model.name):
        print("Train accuracy:\n", ','.join(train_acc_list))
        print("Test accuracy:\n", ','.join(test_acc_list))
    
    print("Best accuracy: ", best_accuracy)

    interval = interval95( best_acc / 100, len(test_data))
    print("Confidence interval (95%):")
    print("[",best_acc - interval[0], best_acc + interval[1], "]" )
    

if __name__ == "__main__":
    main()