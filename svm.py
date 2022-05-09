# Support vector machines
from sklearn.svm import SVC

# Sklearn utils
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# PIL   
from PIL import Image

# Argument parser
import argparse as arg

# Utils
from utils import compute_and_plot_stats
from tqdm import tqdm
import numpy as np
from glob import glob
import pickle

# Parser
parser = arg.ArgumentParser(description= 'Train or test a SVM with the breast cancer dataset.')

# Path to data
parser.add_argument('-d', '--data', dest = 'path', default = 'data', type=str, help= 'Path to dataset.')

# C - Value
parser.add_argument('-c', '--cvalue', dest = 'c_value', default = 1.0, type=float, help = 'Regularization parameter. Default = 1.0.')

# Kernel type
parser.add_argument('-k', '--kernel', dest = 'kernel', default = 'rbf', type=str, help = 'Kernel type, default = "rbf". ')

# Degree (only for kernel function 'poly')
parser.add_argument('-dg', '--degree', dest = 'degree', default = 3, type=int, help = 'Degree of the polynomial kernel function "poly". Not compatible with other kernels. ')

# Gamma
parser.add_argument('-g', '--gamma', dest = 'gamma', default = 'scale', type=str, help = 'Kernel coefficient for "rbf", "poly" and "sigmoid". ')

# Name used for the files generated as output (plots)
parser.add_argument('-na', '--name', dest = 'file_name',default="output", type=str, help= 'Name used for the files generated as output (plots).')

# Test the SVM model
parser.add_argument('-t', '--test', action= 'store_true',dest = 'test',default=False, help= 'Test the SVM model. "-d" argument must be to the path to the test data. "-n" must be the path to model file')

# Enable verbose output
parser.add_argument('-v', '--verbose', action= 'store_true',dest = 'verbose',default=False, help= 'Enable verbose output.')

# ---------- Global variables -----------

# C - Value, Kernel type, Degree, Gamma and Verbose
c_value, kernel, degree, gamma, file_name, verbose = None, None, None, None, None, None

def train(args) -> None:
    """
    Trains a SVC model.

    Args:
        args: The arguments
    """    

    global c_value, kernel, degree, gamma, file_name, verbose

    c_value = args.c_value
    kernel = args.kernel
    degree = args.degree
    gamma = args.gamma
    file_name = args.file_name
    verbose = args.verbose

    # Get data paths
    train_paths = glob(args.path + 'train/*') 
    val_paths = glob(args.path + 'validation/*')

    # Matrix with data samples
    train_data = np.zeros((len(train_paths),7500))
    labels = np.zeros(len(train_paths))

    for i in tqdm(range(len(train_paths)), desc = 'Loading training data...'):

        # Get path
        path = train_paths[i]

        # Get label
        labels[i] = int(path[-5])

        # Open image and convert
        img = Image.open(path)
        img = np.array(img)
        img = img.flatten()

        # Append to training data
        train_data[i] = img

    # Make SVM model in a pipeline
    clf = make_pipeline(StandardScaler(), SVC(C = c_value, kernel = kernel, degree = degree, gamma = gamma, verbose = verbose))
    print("Training SVM...")
    clf.fit(train_data, labels)
    print("Done.")

    # Delete training data from memory
    del train_data
    del labels
    
    # Matrix with validation samples
    true_labels = np.zeros(len(val_paths))
    val_data = np.zeros((len(val_paths),7500))

    for i in tqdm(range(len(val_paths)), desc = 'Loading validation data...'):

        # Get path
        path = val_paths[i]

        # Get label
        true_labels[i] = int(path[-5])

        # Open image and convert
        img = Image.open(path)
        img = np.array(img)
        img = img.flatten()

        # Append to training data
        val_data[i] = img

    # Get predicted labels
    predicted_labels = clf.predict(val_data)

    # Delete validation data from memory
    del val_data

    # Compute and plot metrics
    precision, recall, specificity, f_score, bac = compute_and_plot_stats(true_labels, predicted_labels, file_name)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Specificity:", specificity)
    print("F - Score:", f_score)
    print("Balanced Accuracy:", bac)

    # Confidence interval
    interval = interval95( bac / 100, len(val_paths))
    print("Confidence interval (95%):")
    print(str(bac) + ' +- ' + str(interval * 100))

    # Save model
    print("Saving SVM model...")
    pickle.dump(clf, open(file_name + '.p', 'wb' ))
    print("Done.")

def test(args) -> None:
    """
    Tests the SVC model

    Args:
        args: The arguments.
    """    

    global file_name

    file_name = args.file_name

    # Paths to test data
    test_paths = glob(args.data + '*')

    # Load SVM
    svm = pickle.load(open(file_name,'rb'))

    # Matrix with validation samples
    true_labels = np.zeros(len(test_paths))
    test_data = np.zeros((len(test_paths),7500))

    for i in tqdm(range(len(test_paths)), desc = 'Loading validation data...'):

        # Get path
        path = test_paths[i]

        # Get label
        true_labels[i] = int(path[-5])

        # Open image and convert
        img = Image.open(path)
        img = np.array(img)
        img = img.flatten()

        # Append to training data
        test_data[i] = img

    # Get predicted labels
    predicted_labels = clf.predict(test_data)

    # Compute and plot metrics
    precision, recall, specificity, f_score, bac = compute_and_plot_stats(true_labels, predicted_labels, file_name)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Specificity:", specificity)
    print("F - Score:", f_score)
    print("Balanced Accuracy:", bac)

    # Confidence interval
    interval = interval95( bac / 100, len(val_paths))
    print("Confidence interval (95%):")
    print(str(bac) + ' +- ' + str(interval * 100))


def main():

    # Parse arguments
    args = parser.parse_args()

    if not args.test:
        # Train support vector machine
        train(args)
    else:
        test(args)
    

if __name__ == "__main__":
    main()