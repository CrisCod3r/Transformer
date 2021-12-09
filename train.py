import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from dataset import BreastCancerDataset

from models.AlexNet import *
from models.EfficientNet import *
from models.GoogLeNet import *
from models.LeNet5 import *
from models.ResNet import *
from models.VGG import *

from densenet import *

from torch import Generator, optim, device
from torch.utils.data import Dataset, DataLoader, random_split


import argparse as arg
import sys


def train_cnn(model,train_loader,num_epochs, learning_rate):
    

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 32 == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

    print('Finished training')
    PATH = './cnn.pth'
    torch.save(model.state_dict(), args.file)

def accuracy(model, test_loader):

    model.eval()
    classes = ('benign','malignant')
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(2)]
        n_class_samples = [0 for i in range(2)]

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
            
            for i in range(len(labels)):
                label = labels[i]
                pred = predicted[i]
                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network: {acc} %')

        for i in range(2):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of {classes[i]}: {acc} %')



parser = arg.ArgumentParser(description= 'Train a CNN with the breast cancer dataset.')

# Path to data
parser.add_argument('-d', '--data', dest = 'path', default = 'data', type=str, help= 'Path to dataset')

# Neural network
parser.add_argument('-n', '--net', dest = 'net', default = 'cnn', type=str, help= 'Neural network to train')

# Number of epochs
parser.add_argument('-e', '--epochs', dest= 'num_epochs', default=1, type=int, help= "Number of epochs in training")

# Batch size
parser.add_argument('-b', '--batch_size',dest = 'batch_size', default= 4, type=int, help= 'Batch size')

# Learning rate
parser.add_argument('-lr', '--learning_rate', dest = 'learning_rate', default = 1e-4, type=float, help= 'Start learning rate')

# File in which the hyper parameters will be saved
parser.add_argument('-f', '--file', dest = 'file', default = 'results.pth', type=str, help= 'File where hyper parameters will be saved')
args= parser.parse_args()


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = args.net.lower()

if net == 'alexnet':
    model = AlexNet().to(device)
    model.train()

elif net == 'densenet':
    model = models.densenet161().to(device)
    model.train()

elif net == 'efficientnet':
    model = EfficientNet("b7", num_classes=2).to(device)
    model.train()

elif net == "gdnet":
    model = GDNet().to(device)
    model.train()

elif net == 'inception':
    model = InceptionNet().to(device)
    model.train()

elif net == 'lenet':
    model = LeNet5().to(device)
    model.train()

elif net == 'resnet':
    model = ResNet50().to(device)
    model.train()

elif net == 'vgg':
    model = VGG(vgg_type="VGG16").to(device)
    model.train()
else:
    print("Error, unrecognized neural network")
    print("Available models:", ', '.join(['alexnet', 'densenet', 'efficientnet', "gdnet", 'inception', 'lenet', 'resnet', 'vgg']))
    sys.exit(-1)



print("Loading datasets...")
# Get training dataset (136000 images)
training_data = BreastCancerDataset(args.path)

# A manual seed is introduced to allow the results to be reproduced, remove this parameter if you don't want this to happen
#lengths= [122400, 13600]
training_data, val_data = random_split(training_data, lengths= [122400, 13600], generator=Generator().manual_seed(0))
print("Done")

train_loader = DataLoader(dataset = training_data, batch_size = args.batch_size, shuffle = True)
val_loader = DataLoader(dataset = val_data, batch_size = args.batch_size, shuffle = False)


print("Starting training...")
train_cnn(model,train_loader, args.num_epochs, args.learning_rate)
print("Calculating accuracy...")
accuracy(model, val_loader)