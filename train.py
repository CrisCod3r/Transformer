#PyTorch
import torch
import torch.nn.functional as F

#Utils
from utils import progress_bar
import os

# Others
from typing import Tuple

# Training function
def train(criterion: torch.nn.modules.loss.BCEWithLogitsLoss, device: str, epoch: int, model: torch.nn.Module, model_name: str,  optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.ExponentialLR, trainloader: torch.utils.data.DataLoader) -> None:
    """
    Trains the model for 1 epoch

    Args:
        criterion (torch.nn.modules.loss.BCEWithLogitsLoss): Loss function. Since in this code only uses the BCEWithLogitsLoss, no other loss function is allowed.
        You may modify the code if you need.
        device (str): Device to use (CPU or GPU).
        epoch (int): Current epoch number.
        model (torch.nn.Module): Model to train.
        model_name (str): Name of the model
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

    if not isinstance(criterion, torch.nn.modules.loss.BCEWithLogitsLoss): raise TypeError('"criterion" must be a loss function.')
    if not isinstance(device, str): raise TypeError('"device" must be a str (CPU or GPU).')
    if not isinstance(epoch, int): raise TypeError('"epoch" must be an integer.')
    if not isinstance(model, torch.nn.Module): raise TypeError('"model" must be a torch.nn.Module.')
    if not isinstance(model_name, str): raise TypeError('"model_name" must be a str')
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
        labels = labels.float()

        # Reset gradient
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)[:,:1].squeeze(1)

        # Loss function
        loss = criterion(outputs, labels)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        # Accumulate loss
        train_loss += loss.item()

        # Get predicted output
        probs = torch.sigmoid(outputs) > 0.5
        correct += probs.eq(labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
      
    # Decay Learning Rate
    scheduler.step()
    
    return


def test(best_acc: float, criterion: torch.nn.modules.loss.BCEWithLogitsLoss, device: str, epoch: int, file_name: str, model: torch.nn.Module, model_name: str, testloader: torch.utils.data.DataLoader) -> Tuple[float, list]:
    """
    Test the model

    Args:
        best_acc (float): Best accuracy obtained previously.
        criterion (torch.nn.modules.loss.BCEWithLogitsLoss): Loss function. Since in this code only uses the BCEWithLogitsLoss, no other loss function is allowed.
        You may modify the code if you need.
        device (str): Device to use (CPU or GPU).
        epoch (int): Current epoch number.
        file_name (str): Name of the file where the model weights will be stored.
        model (torch.nn.Module): Model to test.
        model_name (str): Name of the model
        testloader (torch.utils.data.DataLoader): Test data loader.

    Raises:
        TypeError: The given best accuracy is not a float number
        TypeError: The given loss function is not a function
        TypeError: The given device is not a string
        TypeError: The given epoch number is not a integer
        TypeError: The given file name is not a string
        TypeError: The model is not a nn.Module
        TypeError: The given model name is not a string
        TypeError: The given testloader is not DataLoader
    
    Returns:
        float: The accuracy of the model
        list: The accuracy of the model per class
    """    
    if not isinstance(best_acc, float): raise TypeError('"best_acc" must be a float.')
    if not isinstance(criterion, torch.nn.modules.loss.BCEWithLogitsLoss): raise TypeError('"criterion" must be a loss function.')
    if not isinstance(device, str): raise TypeError('"device" must be a str (CPU or GPU).')
    if not isinstance(epoch, int): raise TypeError('"epoch" must be an integer.')
    if not isinstance(file_name, str): raise TypeError('"file_name" must be a string.')
    if not isinstance(model, torch.nn.Module): raise TypeError('"model" must be a torch.nn.Module.')
    if not isinstance(model_name, str): raise TypeError('"model_name" must be a string.')
    if not isinstance(testloader, torch.utils.data.DataLoader): raise TypeError('"testloader" must be a torch.utils.data.DataLoader')

    #Set model to evaluation
    model.eval()

    test_loss = 0
    correct = 0
    total = 0

    # Disable gradients
    with torch.no_grad():

        for batch_idx, (inputs, labels) in enumerate(testloader):

            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.float()
            
            # Forward pass
            outputs = model(inputs)[:,:1].squeeze(1)

            # Loss function
            loss = criterion(outputs, labels)

            # Accumulate test loss
            test_loss += loss.item()
            
            # Get predicted output
            probs = torch.sigmoid(outputs) > 0.5
            correct += probs.eq(labels).sum().item()
            total += labels.size(0)
            

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
   

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving checkpoint...')
        state = {
            'model': model.state_dict(),
            'accuracy': acc,
            'epoch': epoch
        }

        if not os.path.isdir('pretrained'):
            os.mkdir('pretrained')
        torch.save(state, './pretrained/' + file_name + '.pth')


    # Return this epoch's test loss, test accuracy and class accuracy
    # return test_loss / (batch_idx+1), acc, [100.0 * n_class_correct[i] / n_class_samples[i] for i in range(len(classes))]
    return acc

def predict(device: str, model: torch.nn.Module, model_name: str, testloader: torch.utils.data.DataLoader) -> Tuple[list, list]:
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

    if not isinstance(device, str): raise TypeError('"device" must be a string (CPU or GPU).')
    if not isinstance(model, torch.nn.Module): raise TypeError('"model" must be a torch.nn.Module.')
    if not isinstance(model_name, str): raise TypeError('"model_name" must be a string.')
    if not isinstance(testloader, torch.utils.data.DataLoader): raise TypeError('"testloader" must be a torch.utils.data.DataLoader')
    
    #Set model to evaluation
    model.eval()

    correct = 0
    total = 0


    true_labels, predicted_labels, probabilities = [], [], []

    with torch.no_grad():

        for batch_idx, (inputs, labels) in enumerate(testloader):

            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.float()
            
            # Forward pass
            outputs = model(inputs)[:,:1].squeeze(1)

            # Get predicted probability 
            probs = torch.sigmoid(outputs).tolist()

            # Convert probability to class
            predicted_class = torch.Tensor([1 if elem > 0.5 else 0 for elem in probs]).to(device)

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