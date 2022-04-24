from utils import progress_bar
import torch
import os

# Training function
def train(criterion, device, epoch, model, optimizer, scheduler, trainloader):
    """
    Trains the network for 1 epoch
    Args:
        criterion: Loss criterion
        epoch: Epoch #
        model: Network to train
        optimizer: Learning rate optimizer
        trainloader: Train data
    """
    
    print('-----=| Epoch %d |=-----' % epoch)

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
        
        if model.name == "WeightedNet":
            model.update_weights(labels)
        # Update progress bar
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Decay Learning Rate
    scheduler.step()
    
    # Return this epoch's loss and train accuracy
    # return train_loss / (batch_idx+1), 100.*correct/total
    return


def test(best_acc, classes, criterion, device, epoch, model, optimizer, testloader):

    #Set model to evaluation
    model.eval()

    test_loss = 0
    correct = 0
    total = 0

    n_class_correct = [0 for i in range(len(classes))]
    n_class_samples = [0 for i in range(len(classes))]

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
        if model.name == "WeightedNet":

            state = {
                'model': model.state_dict(),
                'accuracy': acc,
                'class_accuracy': [100.0 * n_class_correct[i] / n_class_samples[i] for i in range(len(classes))],
                'epoch': epoch,
                'weights': model.weights()
            }
        else:
            state = {
            'model': model.state_dict(),
            'accuracy': acc,
            'class_accuracy': [100.0 * n_class_correct[i] / n_class_samples[i] for i in range(len(classes))],
            'epoch': epoch

        }
        if not os.path.isdir('pretrained'):
            os.mkdir('pretrained')
        torch.save(state, './pretrained/' + model.name + '.pth')


    # Return this epoch's test loss, test accuracy and class accuracy
    # return test_loss / (batch_idx+1), acc, [100.0 * n_class_correct[i] / n_class_samples[i] for i in range(len(classes))]
    return acc, [100.0 * n_class_correct[i] / n_class_samples[i] for i in range(len(classes))]


def final_test(classes, device, model, testloader):

    #Set model to evaluation
    model.eval()

    correct = 0
    total = 0

    n_class_correct = [0 for i in range(len(classes))]
    n_class_samples = [0 for i in range(len(classes))]

    with torch.no_grad():

        for batch_idx, (inputs, labels) in enumerate(testloader):

            inputs, labels = inputs.to(device), labels.to(device)

            # Get predicted output
            outputs = model(inputs)
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
    
    # Return this epoch's test loss, test accuracy and class accuracy
    # return test_loss / (batch_idx+1), acc, [100.0 * n_class_correct[i] / n_class_samples[i] for i in range(len(classes))]
    return [100.0 * n_class_correct[i] / n_class_samples[i] for i in range(len(classes))]

def test_and_return(device, model, testloader):

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