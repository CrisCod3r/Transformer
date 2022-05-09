#PyTorch
import torch

#Utils
from utils import progress_bar
import os

# Training function
def train(criterion: torch.nn.modules.loss.CrossEntropyLoss, device: str, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.ExponentialLR, trainloader: torch.utils.data.DataLoader) -> None:
    """
    Trains the model for 1 epoch

    Args:
        criterion (torch.nn.modules.loss.CrossEntropyLoss): Loss function. Since in this code only uses the CrossEntropyLoss, no other loss function is allowed.
        You may modify the code if you need.
        device (str): Device to use (CPU or GPU).
        epoch (int): Current epoch number.
        model (torch.nn.Module): Model to train.
        optimizer (torch.optim): Optimizer to use.
        scheduler (torch.optim.lr_scheduler.ExponentialLR): Learning rate scheduler. Only ExponentialLR is used. You may modify the code if you want.
        trainloader (torch.utils.data.DataLoader): Training data loader.

    Raises:
        TypeError: The given loss function is not a function
        TypeError: The given device is not a str
        TypeError: The given epoch number is not a integer
        TypeError: The model is not a nn.Module
        TypeError: The given optimizer is not a learning rate optimizer
        TypeError: The given scheduler is not a learning rate scheduler
        TypeError: The given trainloader is not DataLoader.
    
    Returns:
        None
    """    

    if not isinstance(criterion, torch.nn.modules.loss.CrossEntropyLoss): raise TypeError('"criterion" must be a loss function.')
    if not isinstance(device, str): raise TypeError('"device" must be a str (CPU or GPU).')
    if not isinstance(epoch, int): raise TypeError('"epoch" must be an integer.')
    if not isinstance(model, torch.nn.Module): raise TypeError('"model" must be a torch.nn.Module.')
    if not isinstance(optimizer, torch.optim.Optimizer): raise TypeError('"optimizer" must be a torch.optim.Optimizer.')
    if not isinstance(scheduler, torch.optim.lr_scheduler.ExponentialLR): raise TypeError('"scheduler" must be a torch.optim.lr_scheduler.')
    if not isinstance(trainloader, torch.utils.data.DataLoader): raise TypeError('"trainloader" must be a torch.utils.data.DataLoader')
    
    print('-------=| Epoch %d |=-------' % epoch)

    # Set model to train
    model.train()

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(trainloader):

        inputs, labels = inputs.to(device), labels.to(device)

        # Reset gradient
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        loss.backward()
        optimizer.step()

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
    
    # Return this epoch's loss and train accuracy
    # return train_loss / (batch_idx+1), 100.*correct/total
    return


def test(best_acc: float, classes: list, criterion: torch.nn.modules.loss.CrossEntropyLoss, device: str, epoch: int, file_name: str, model: torch.nn.Module, testloader: torch.utils.data.DataLoader) -> [float, list]:
    """
    Test the model

    Args:
        best_acc (float): Best accuracy obtained previously.
        classes (list): List of strings indicating the name of the classes.
        criterion (torch.nn.modules.loss.CrossEntropyLoss): Loss function. Since in this code only uses the CrossEntropyLoss, no other loss function is allowed.
        You may modify the code if you need.
        device (str): Device to use (CPU or GPU).
        epoch (int): Current epoch number.
        file_name (str): Name of the file where the model weights will be stored.
        model (torch.nn.Module): Model to test.
        testloader (torch.utils.data.DataLoader): Test data loader.

    Raises:
        TypeError: The given best accuracy is not a float number
        TypeError: The given list of classes is not a list
        TypeError: The given loss function is not a function
        TypeError: The given device is not a str
        TypeError: The given epoch number is not a integer
        TypeError: The given file name is not a str
        TypeError: The model is not a nn.Module
        TypeError: The given testloader is not DataLoader.
    
    Returns:
        None
    """    
    if not isinstance(best_acc, float): raise TypeError('"best_acc" must be a float.')
    if not isinstance(classes, list): raise TypeError('"classes" must be a list of strings.')
    if not isinstance(criterion, torch.nn.modules.loss.CrossEntropyLoss): raise TypeError('"criterion" must be a loss function.')
    if not isinstance(device, str): raise TypeError('"device" must be a str (CPU or GPU).')
    if not isinstance(epoch, int): raise TypeError('"epoch" must be an integer.')
    if not isinstance(file_name, str): raise TypeError('"file_name" must be a string.')
    if not isinstance(model, torch.nn.Module): raise TypeError('"model" must be a torch.nn.Module.')
    if not isinstance(testloader, torch.utils.data.DataLoader): raise TypeError('"testloader" must be a torch.utils.data.DataLoader')

    #Set model to evaluation
    model.eval()

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
            outputs = model(inputs)
            loss = criterion(outputs, labels)

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
    if acc > best_acc:
        print('Saving checkpoint...')
        state = {
            'model': model.state_dict(),
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

def test_and_return(device: str, model: torch.nn.Module, testloader: torch.utils.data.DataLoader) -> [list, list]:
    """
    Uses the model to classify the images in the given testloader.

    Args:
        device (str): Device to use (CPU or GPU)
        model (torch.nn.Module): The model to use for predicting
        testloader (torch.utils.data.DataLoader): Data loader

    Raises:
        TypeError: The given dice is not a str
        TypeError: The given model is not a torch.nn.Module
        TypeError: The given testloader is not a torch.utisl.data.DataLoader

    Returns:
        list: The true labels of the images.
        list: The predicted labels of the images.
    """    

    if not isinstance(device, str): raise TypeError('"device" must be a str (CPU or GPU).')
    if not isinstance(model, torch.nn.Module): raise TypeError('"model" must be a torch.nn.Module.')
    if not isinstance(testloader, torch.utils.data.DataLoader): raise TypeError('"testloader" must be a torch.utils.data.DataLoader')
    #Set model to evaluation
    model.eval()

    correct = 0
    total = 0


    true_labels, predicted_labels = [], []

    with torch.no_grad():

        for batch_idx, (inputs, labels) in enumerate(testloader):

            inputs, labels = inputs.to(device), labels.to(device)

            # Get predicted output
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            # Accumulate true and predicted labels
            true_labels.extend( labels.data.tolist() )
            predicted_labels.extend( predicted.tolist() )

            # Add total and correct predictions
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            progress_bar(batch_idx, len(testloader), ' Acc: %.3f%% (%d/%d)'
                % (100.*correct/total, correct, total))
    
    return true_labels, predicted_labels