# PyTorch
import torch
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
import torchvision.transforms as transforms
# Dataset
from dataset import BreastCancerDataset

# Utils
from utils import interval95, compute_and_plot_stats, build_optimizer, load_pca_matrix, progress_bar
import transformers

# Others
import argparse as arg
import os
from typing import Tuple

parser = arg.ArgumentParser(description= 'Train or test a ViT with the breast cancer dataset.')

# Path to training data
parser.add_argument('-tr', '--training_path', dest = 'training_path', default = None, type=str, help= 'Path to training dataset.')

# Path to test data
parser.add_argument('-te', '--test_path', dest = 'test_path', default = None, type=str, help= 'Path to test dataset (can also be path to validation data).')

# Name of the ViT
parser.add_argument('-vt', '--vit', dest = 'vit_name', default = None, type=str, help= 'Name of the ViT.')

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
parser.add_argument('-na', '--name', dest = 'file_name', default="output", type=str, help= 'Name used for the files generated as output (plots)')

# Number of components for PCA projection
parser.add_argument('-p', '--pca', dest = 'pca_components', default=None, type=int, help= 'Number of components for PCA projection. If not used, no PCA projection will be applied')

# -----------=| Global variables |=---------------------
model = None

best_accuracy = 0.0

file_name = None

training_data, test_data = None, None

device, criterion, optimizer, scheduler, pca = None, None, None, None, None


def build_model(model_name: str) -> Tuple[transformers.FeatureExtractionMixin, torch.nn.Module]:
    """
    Builds the model given its name

    Args:
        model_name (str): The name of the ViT

    Returns:
        Tuple[transformers.FeatureExtractionMixin, torch.nn.Module]: The feature extractor and the model
    """    

    if not isinstance(model_name, str): raise TypeError('"model_name" must be a string')

    # Convert to lower case
    model_name = model_name.lower()

    if model_name == "deit":
        feature_extractor = transformers.DeiTFeatureExtractor.from_pretrained('facebook/deit-base-distilled-patch16-224')
        model = transformers.DeiTForImageClassification.from_pretrained('facebook/deit-base-distilled-patch16-224')

    if model_name == "vit":
        feature_extractor = transformers.ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        model = transformers.ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    return feature_extractor, model


def train(epoch: int, feature_extractor, model) -> None:
    """
    Trains the model for 1 epoch

    Args:
        epoch (int): The current epoch

    Returns:
        None
    """   

    
    global device, criterion, optimizer, scheduler, training_data

    print('-------=| Epoch %d |=-------' % epoch)

    # Set model to train
    model.train()

    train_loss = 0
    correct = 0
    total = 0

    for idx in range(len(training_data)):
        
        # Get image
        img, label = training_data.__getitem__(idx)

        # Transform label (int) to tensor for computing
        label = torch.tensor([label],device=torch.device('cuda:0')) if torch.cuda.is_available() else torch.tensor([label])

        # Extract features
        inputs = feature_extractor(img, return_tensors = 'pt')
        inputs.to(device)

        # Reset gradient
        optimizer.zero_grad()

        # Forward pass
        outputs = model(**inputs).logits

        # Backward and optimize
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        # Accumulate loss
        train_loss += loss.item()

        # Get predicted output
        predicted = outputs.argmax(-1)
        total += label.size(0)
        correct += predicted.eq(label).sum().item()
        
        # Update progress bar
        progress_bar(idx, len(training_data), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss/(idx+1), 100.*correct/total, correct, total))
      
    # Decay Learning Rate
    scheduler.step()

    return

def test(epoch: int,feature_extractor, model):
    """
    Tests the BeiT

    Args:
        epoch (int): The epoch number.

    Returns:
        Tuple[float, list]: Accuracy obtained and per class accuracies.
    """   
    
    global best_accuracy, device, criterion, file_name, optimizer, scheduler, test_data
    
    #Set model to evaluation
    model.eval()

    test_loss = 0
    correct = 0
    total = 0

    # Disable gradients
    with torch.no_grad():

        for idx in range(len(test_data)):
        
            # Get image
            img, label = test_data.__getitem__(idx)
                       
            # Transform label (int) to tensor for computing
            label = torch.tensor([label],device=torch.device('cuda:0')) if torch.cuda.is_available() else torch.tensor([label])

            # Extract features
            inputs = feature_extractor(img, return_tensors = 'pt')
            inputs.to(device)

            # Forward pass
            outputs = model(**inputs).logits

            # Backward and optimize
            loss = criterion(outputs, label)

            # Accumulate loss
            test_loss += loss.item()

            # Get predicted output
            predicted = outputs.argmax(-1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            
            # Update progress bar
            progress_bar(idx, len(test_data), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(idx+1), 100.*correct/total, correct, total))
   

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_accuracy:
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

def predict(feature_extractor, model):

    global best_accuracy, device, criterion, optimizer, scheduler, testloader

    #Set model to evaluation
    model.eval()

    correct = 0
    total = 0


    true_labels, predicted_labels, probabilities = [], [], []

    with torch.no_grad():

        for idx in range(len(test_data)):
        
            # Get image
            img, label = test_data.__getitem__(idx)

            # Transform label (int) to tensor for computing
            labels = torch.tensor([label],device=torch.device('cuda:0')) if torch.cuda.is_available() else torch.tensor([label])

            # Extract features
            inputs = feature_extractor(img, return_tensors = 'pt')
            inputs.to(device)

            # Forward pass
            outputs = model(**inputs).logits

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
            progress_bar(idx, len(test_data), ' Acc: %.3f%% (%d/%d)'
                % (100.*correct/total, correct, total))


    
    return true_labels, predicted_labels, probabilities

def main():

    global model, best_accuracy, device, criterion, optimizer, file_name, pca, scheduler, training_data, test_data

    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Criterion
    criterion = torch.nn.CrossEntropyLoss()

    # Parse args
    args = parser.parse_args()

    # Obtain output file name
    file_name = args.file_name
    
    # Obtain number of components for PCA dimensionality reduction
    pca_components = args.pca_components
   
    if os.path.isfile('./pretrained/' + file_name + '.pth'):
        print("Previous training with this models found. Obtaining best accuracy...")
        state_dict = torch.load('./pretrained/' + file_name + '.pth')
        best_accuracy = state_dict['accuracy']
        print("Best accuracy: ", best_accuracy)

    if args.resume:
        state_dict = torch.load('./pretrained/' + file_name + '.pth')
        model.load_state_dict( state_dict['model'] )
        best_accuracy = state_dict['accuracy']
        print("Loaded checkpoint, best accuracy obtained previously is: %.3f" % best_accuracy)

    feature_extractor, model = build_model(args.vit_name)

    model.to(device)

    # Build optimizer
    optimizer = args.optimizer.lower()
    optimizer = build_optimizer(model, optimizer) 

    # Build scheduler
    scheduler = ExponentialLR(optimizer, gamma = 0.95, verbose=True)

    # Data augmentation
    print("Loading data augmentation transforms...")

    train_transform = transforms.Compose([
        # Convert to Tensor
        transforms.ToTensor(),

        # Normalize train dataset with its mean and standard deviation
        transforms.Normalize((0.7595, 0.5646, 0.6882), (0.1496, 0.1970, 0.1428)),

        # Since the models only take PIL image this must be the last transform (see Huggingface docs)
        transforms.ToPILImage(),

        # Allow random horizontal flips (data augmentation)
        transforms.RandomHorizontalFlip(p = 0.25),

        # Allow random vertical flips (data augmentation)
        transforms.RandomVerticalFlip(p = 0.05),
        
        # This resize is required to provide a correct input 
        # to a pretrained ViT of patch size == 16 or patch size == 32
        transforms.Resize((224,224))


    ])

    test_transform = transforms.Compose([

        # Convert to Tensor
        transforms.ToTensor(),

        # Normalize test dataset with the mean and standard deviation of the training data
        transforms.Normalize((0.7595, 0.5646, 0.6882), (0.1497, 0.1970, 0.1428)),

        # Since the models only take PIL image this must be the last transform (see Huggingface docs)
        transforms.ToPILImage(),

        # This resize is required to provide a correct input 
        # to a pretrained ViT of patch size == 16 or patch size == 32
        transforms.Resize((224,224))


    ])

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

    best_accuracy = 0.0
    for epoch in range(args.num_epochs):
        
        # Train for 1 epoch
        train(epoch, feature_extractor, model)

        # Test the model
        test_acc = test(epoch, feature_extractor, model)

        if test_acc > best_accuracy:
            best_accuracy = test_acc  


    #  Get model when it had the best accuracy
    
    # Obtain predictions
    print("Obtaining predictions...")
    true_labels, predicted_labels, probabilities = predict(feature_extractor, model)

    # Compute and plot metrics
    precision, recall, specificity, f_score, bac = compute_and_plot_stats(true_labels, predicted_labels, probabilities, file_name)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Specificity:", specificity)
    print("F - Score:", f_score)
    print("Balanced Accuracy:", bac)
    print("Best accuracy: ", best_accuracy)

    # Confidence interval
    interval = interval95( best_accuracy / 100, test_samples)
    print("Confidence interval (95%):")
    print(str(best_accuracy) + ' +- ' + str(interval * 100))

    

if __name__ == "__main__":
    main()
