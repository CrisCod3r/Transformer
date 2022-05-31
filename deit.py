# PyTorch
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import  DataLoader
from models.IDCNet import *
import torch.nn.functional as F

# Dataset
from dataset import BreastCancerDataset

# Utils
from utils import interval95, compute_and_plot_stats, build_optimizer, build_transforms, load_pca_matrix, progress_bar
from vit_pytorch.distill import DistillableViT, DistillWrapper

# Others
import argparse as arg
import os
from typing import Tuple

parser = arg.ArgumentParser(description= 'Train or test a ViT with the breast cancer dataset.')

# Path to training data
parser.add_argument('-tr', '--training_path', dest = 'training_path', default = None, type=str, help= 'Path to training dataset.')

# Path to test data
parser.add_argument('-te', '--test_path', dest = 'test_path', default = None, type=str, help= 'Path to test dataset (can also be path to validation data).')


# Number of epochs
parser.add_argument('-e', '--epochs', dest= 'num_epochs', default= 1, type=int, help= "Number of epochs in training")

# Batch size
parser.add_argument('-b', '--batch_size',dest = 'batch_size', default= 8, type=int, help= 'Batch size')

# Optimizer
parser.add_argument('-o', '--optimizer', dest = 'optimizer', default="adam", type=str, help= 'Learning rate optimizer')

# Resume training from checkpoint
parser.add_argument('-r', '--resume', action= 'store_true', dest = 'resume', default=False, help= 'Resume training from checkpoint')

# Test the neural network (requiers the -n parameter)
parser.add_argument('-t', '--test', action= 'store_true', dest = 'test', default=False, help= 'Test a ViT (requiers the -n and -te parameter)')

# Name used for the files generated as output (plots)
parser.add_argument('-na', '--name', dest = 'file_name', default="DeiT", type=str, help= 'Name used for the files generated as output (plots)')

# Number of components for PCA projection
parser.add_argument('-p', '--pca', dest = 'pca_components', default=None, type=int, help= 'Number of components for PCA projection. If not used, no PCA projection will be applied')

# -----------=| Global variables |=---------------------
DeiT, distiller = None, None

best_accuracy = 0.0
best_class_accuracy = []
classes = ['benign','malignant']

file_name = None
device, criterion, optimizer, scheduler, trainloader, testloader, pca = None, None, None, None, None, None, None

def train(epoch: int) -> None:
    """
    Trains the model for 1 epoch

    Args:
        epoch (int): The current epoch

    Returns:
        None
    """    

    
    global criterion, DeiT, device, distiller, optimizer, scheduler, trainloader

    print('-------=| Epoch %d |=-------' % epoch)

    # Set model to train
    distiller.train()

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(trainloader):

        inputs, labels = inputs.to(device), labels.to(device)

        # Reset gradient
        optimizer.zero_grad()

        # Backward and optimize
        loss = distiller(inputs, labels)
        loss.backward()
        optimizer.step()

        outputs = DeiT(inputs)

        # Accumulate loss
        train_loss += loss.item()

        # Get predicted output
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
      
    # Decay Learning Rate
    scheduler.step()

    return

def test(epoch: int) -> Tuple[float, list]:
    """
    Tests the DeiT

    Args:
        epoch (int): The epoch number.

    Returns:
        Tuple[float, list]: Accuracy obtained and per class accuracies.
    """    
    
    global best_accuracy, best_class_accuracy, classes, device, criterion, file_name, optimizer, scheduler, testloader
    
    #Set model to evaluation
    distiller.eval()

    test_loss = 0
    correct = 0
    total = 0

    n_class_correct = [0 for i in range(len(classes))]
    n_class_samples = [0 for i in range(len(classes))]

    # Disable gradients
    with torch.no_grad():

        for batch_idx, (inputs, labels) in enumerate(testloader):

            inputs, labels = inputs.to(device), labels.to(device)

            # Get predicted output
            outputs = DeiT(inputs)
            loss = distiller(inputs, labels)

            # Accumulate test loss
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Accuracy per class
            for i in range(len(labels)):

                label = labels[i]
                pred = predicted[i]

                if (label == pred):
                    n_class_correct[label] += 1

                n_class_samples[label] += 1

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
   

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_accuracy:
        print('Saving checkpoint...')
        state = {
            'model': DeiT.state_dict(),
            'accuracy': acc,
            'class_accuracy': [100.0 * n_class_correct[i] / n_class_samples[i] for i in range(len(classes))],
            'epoch': epoch
        }

        if not os.path.isdir('pretrained'):
            os.mkdir('pretrained')
        torch.save(state, './pretrained/' + file_name + '.pth')


    # Return this epoch's test loss, test accuracy and class accuracy
    # return test_loss / (batch_idx+1), acc, [100.0 * n_class_correct[i] / n_class_samples[i] for i in range(len(classes))]
    return acc, [100.0 * n_class_correct[i] / n_class_samples[i] for i in range(len(classes))]

def predict() -> Tuple[list, list, list]:
    """
    Obtains the predicted labels given a tetloader

    Returns:
        Tuple[list, list, list]: The true, predicted and associated probabilites (for ROC-AUC plotting)
    """    
    global  best_accuracy, best_class_accuracy, classes, criterion, DeiT, device, distiller, optimizer, scheduler, testloader

    #Set model to evaluation
    distiller.eval()

    correct = 0
    total = 0


    true_labels, predicted_labels, probabilities = [], [], []

    with torch.no_grad():

        for batch_idx, (inputs, labels) in enumerate(testloader):

            inputs, labels = inputs.to(device), labels.to(device)

            # Get predicted output
            outputs = DeiT(inputs)

            # Get predicted probability and class
            probs = F.softmax(outputs, dim = 1)
            _, predicted_class = probs.topk(1, dim = 1)
            # Since this is a binary classification problem, the metrics used to calculate
            # the ROC-AUC score need a probability, which is the one associated with the prediction.
            # However, it's not always the maximum the probability of both classes, rather always the probability
            # of the same class (even if it's less than the other one), that's why I chose to always take the probability
            # of the malignant class
            probs = [elem[1] for elem in probs.tolist()]

            # Accumulate true, predicted labels and probabilities
            true_labels.extend( labels.data.tolist() )
            predicted_labels.extend( predicted_class.tolist() )
            probabilities.extend( probs )

            # Add total and correct predictions
            total += labels.size(0)
            correct += predicted_class.eq(labels).sum().item()

            # Update progress bar
            progress_bar(batch_idx, len(testloader), ' Acc: %.3f%% (%d/%d)'
                % (100.*correct/total, correct, total))

    
    return true_labels, predicted_labels, probabilities

def main():

    global DeiT, distiller, device, criterion, optimizer, file_name, pca, scheduler, trainloader, testloader

    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Criterion
    criterion = torch.nn.CrossEntropyLoss()
    
    # Parse arguments
    args = parser.parse_args()

    # Obtain output file name
    file_name = args.file_name
    
    # Obtain number of components for PCA dimensionality reduction
    pca_components = args.pca_components

    # Model (this part of the code is manually modified)
    print('Building model...')
    model = IDCNet_tiny(0.8)
    #state_dict = torch.load('./pretrained/ConvNeXt_large.pth')
    #model.load_state_dict( state_dict['model'] )

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

    DeiT = DistillableViT(
        image_size = 50,
        patch_size = 10,
        num_classes = 2,
        dim = 512,
        depth = 3,
        heads = 4,
        mlp_dim = 1024,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    distiller = DistillWrapper(
        student = DeiT,
        teacher = model,
        temperature = 3,           # temperature of distillation
        alpha = 0.5,               # trade between main loss and distillation loss
        hard = True               # whether to use soft or hard distillation
    )

    DeiT.to(device)
    distiller.to(device)

    optimizer = args.optimizer.lower()
    optimizer = build_optimizer(model, optimizer) 

    # Build scheduler
    scheduler = ExponentialLR(optimizer, gamma = 0.95, verbose=True)

    # Data augmentation
    print("Loading data augmentation transforms...")
    train_transform, test_transform = build_transforms('convnext_large', pca_components is not None)

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

    best_accuracy = 0
    classes = ['benign','malignant']
    for epoch in range(args.num_epochs):
        
        # Train 1 epoch
        train(epoch)

        # Test the model
        test_acc, class_accuracy = test(epoch)

        if test_acc > best_accuracy:
            best_accuracy = test_acc 
            best_class_accuracy = class_accuracy   

    del DeiT, distiller

    state_dict = torch.load('./pretrained/' + file_name + '.pth') ['model']
    DeiT = DistillableViT(
        image_size = 50,
        patch_size = 10,
        num_classes = 2,
        dim = 512,
        depth = 3,
        heads = 4,
        mlp_dim = 1024,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    # Load state dict into DeiT
    DeiT.load_state_dict(state_dict)
    distiller = DistillWrapper(
        student = DeiT,
        teacher = model,
        temperature = 3,           # temperature of distillation
        alpha = 0.5,               # trade between main loss and distillation loss
        hard = True               # whether to use soft or hard distillation
    )
    # Obtain predictions
    print("Obtaining predictions...")
    true_labels, predicted_labels, probabilities = predict()

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

    for idx in range(len(classes)):
        print("Accuracy of class " + classes[idx] + ": %.3f" % best_class_accuracy[idx])


if __name__ == "__main__":
    main()