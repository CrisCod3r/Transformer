import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from dataset import BreastCancerDataset
from Convolutional_NN import *

import torch
from torch import Generator, optim, device
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import BreastCancerDataset

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
    torch.save(model.state_dict(), PATH)

def accuracy(model, test_loader):

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
            
            for i in range(batch_size):
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

# Number of epochs
parser.add_argument('-e', '--epochs', dest= 'num_epochs', default=1, type=int, help= "Number of epochs in training")

# Batch size
parser.add_argument('-b', '--batch_size',dest = 'batch_size', default= 4, type=int, help= 'Batch size')

# Learning rate
parser.add_argument('-lr', '--learning_rate', dest = 'learning_rate', default = 0.1, type=float, help= 'Learning rate')

args= parser.parse_args()
print(args.num_epochs, type(args.num_epochs))
print(args.batch_size, type(args.batch_size))
print(args.learning_rate, type(args.learning_rate))
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""# Hyper parameters 
num_epochs = 3
batch_size = 32
learning_rate = 0.1"""

# dataset has PILImage images of range [0, 1]. 
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

print("Loading datasets...")
# Get training dataset (136000 images)
training_data = BreastCancerDataset('../data/train/')

# A manual seed is introduced to allow the results to be reproduced, remove this parameter if you don't want this to happen
training_data, val_data = random_split(training_data, lengths= [122400, 13600], generator=Generator().manual_seed(6))
print("Done")

train_loader = DataLoader(dataset = training_data, batch_size = args.batch_size, shuffle = True)
val_loader = DataLoader(dataset = val_data, batch_size = args.batch_size, shuffle = False)

model = CNN().to(device)
model.train()
print("Starting training...")
train_cnn(model,train_loader, args.num_epochs, args.learning_rate)
print("Calculating accuracy...")
accuracy(model, val_loader)