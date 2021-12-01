from dataset import BreastCancerDataset
from Convolutional_NN import *

import torch
from torch import Generator, optim, device
from torch.utils.data import Dataset, DataLoader, random_split

def load_data(batch_size = 128):

    # Get training dataset (136000 images)
    training_data = BreastCancerDataset('../data/train/')

    # A manual seed is introduced to allow the results to be reproduced, remove this parameter if you don't want this to happen
    training_data, val_data = random_split(training_data, lengths= [122400, 13600], generator=Generator().manual_seed(6))

    train_loader = DataLoader(dataset = training_data, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(dataset = val_data, batch_size = batch_size, shuffle = False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return train_loader, val_loader, device


def train_cnn(model, train_loader, val_loader, device, epochs = 1, lr = 0.1):

    optimizer = torch.optim.Adam(model.parameters(),lr = lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print("Epoch #",epoch)
        model.train()
        for idx, (inputs,labels) in enumerate(train_loader):

            inputs = inputs.to(device)
            labels = labels.to(device)

            # Reset gradient
            optimizer.zero_grad()

            probs = model(inputs)

            # Calculate loss function
            loss = criterion(probs, labels)

            # Propagate error
            loss.backward()

            optimizer.step()

            if idx % 30 == 0:
                print("Batch: %d %d, Loss = %.5f" % (idx, len(train_loader), loss.item() ))
                

    train_accuracy = get_accuracy(model,train_loader,device)
    val_accuracy = get_accuracy(model,train_loader,device)

    print("Accuracy in training: %.3f%%" % (100*train_accuracy))
    print("Accuracy in validation: %.3f%%" % (100*val_accuracy))

def get_accuracy (model, dataloader, device):

    model.eval()
    correct = 0
    samples = 0

    # Disables autograd, speeding up computation and reducing memory
    with torch.no_grad():
        for inputs,labels in dataloader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            # Get the predicted label as the maximum of the softmax probabilities
            predictions = model(data).max(1) [1]
            correct += (predictions == labels).sum().item()
            samples += prediction.shape[0]
    
    return correct / samples

print("Loading data...")
train_loader, val_loader, device = load_data()
print("Done")
model = CNN()
print("Setting up train...")
train_cnn(model, train_loader, val_loader, device, epochs = 1, lr = 0.1)