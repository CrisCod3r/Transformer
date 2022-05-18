# Support vector machines
from sklearn.svm import SVC

# Sklearn utils
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Image managing
from PIL import Image
import cv2

# Argument parser
import argparse as arg

# Utils
from utils import compute_and_plot_stats, interval95, load_pca_matrix
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

# PCA Components
parser.add_argument('-p', '--pca', dest = 'pca_comp', default = None, type=int, help = 'Number of components to use in PCA Projection. If the parameter is ommited, no projection will be done.')

# ---------- Global variables -----------

# C - Value, Kernel type, Degree, Gamma and Verbose
c_value, kernel, degree, gamma, file_name, verbose, pca_comp = None, None, None, None, None, None, None

def train(args) -> None:
    """
    Trains a SVC model.

    Args:
        args: The arguments
    """    

    global c_value, kernel, degree, gamma, file_name, verbose, pca_comp

    c_value = args.c_value
    kernel = args.kernel
    degree = args.degree
    gamma = args.gamma
    file_name = args.file_name
    verbose = args.verbose
    pca_comp = args.pca_comp

    # Get data paths
    train_paths = glob(args.path + 'train/*')
    val_paths = glob(args.path + 'validation/*') 
    # Load PCA matrix
    if pca_comp is not None:
        print("Loading PCA matrix")
        pca = load_pca_matrix(pca_comp)

    # Matrix with data samples
    train_data = np.zeros((len(train_paths),7500))
    labels = np.zeros(len(train_paths))

    if pca_comp is None:
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
        
    else:
        for i in tqdm(range(len(train_paths)), desc = 'Loading training data with PCA...'):

            # Get path
            path = train_paths[i]

            # Get label
            labels[i] = int(path[-5])

            # Load image using OpenCV (not PIL, this is done this way to use PCA correctly)
            img = cv2.cvtColor( cv2.imread(path), cv2.COLOR_BGR2RGB )

            # Split image into RGB channels
            blue, green, red = cv2.split(img)

            # Normalize
            blue, green, red = blue/255, green/255, red/255

            # Project data to lower dimensions
            red = pca['red'].transform([ red.flatten() ])
            green = pca['green'].transform([ green.flatten() ])
            blue = pca['blue'].transform([ blue.flatten() ])

            # Reconstruct data
            red = pca['red'].inverse_transform(red)
            green = pca['green'].inverse_transform(green)
            blue = pca['blue'].inverse_transform(blue)

            # Concatenate channels
            img = np.concatenate((blue,green,red), axis = None)

            # Append to training data
            train_data[i] = img

    # Make SVM model in a pipeline
    svm = make_pipeline(StandardScaler(), SVC(C = c_value, kernel = kernel, degree = degree, gamma = gamma, verbose = verbose, probability = True))

    print("Training SVM...")
    svm.fit(train_data, labels)
    print("Done.")

    # Delete training data from memory
    del train_data
    del labels
    
    # Matrix with validation samples
    true_labels = np.zeros(len(val_paths))
    val_data = np.zeros((len(val_paths),7500))
    probabilities = []

    if pca_comp is None:
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
        
    else:
        for i in tqdm(range(len(val_paths)), desc = 'Loading validation data with PCA...'):

            # Get path
            path = val_paths[i]

            # Get label
            true_labels[i] = int(path[-5])

            # Load image using OpenCV (not PIL, this is done this way to use PCA correctly)
            img = cv2.cvtColor( cv2.imread(path), cv2.COLOR_BGR2RGB )

            # Split image into RGB channels
            blue, green, red = cv2.split(img)

            # Normalize
            blue, green, red = blue/255, green/255, red/255

            # Project data to lower dimensions
            red = pca['red'].transform([ red.flatten() ])
            green = pca['green'].transform([ green.flatten() ])
            blue = pca['blue'].transform([ blue.flatten() ])

            # Reconstruct data
            red = pca['red'].inverse_transform(red)
            green = pca['green'].inverse_transform(green)
            blue = pca['blue'].inverse_transform(blue)

            # Concatenate channels
            img = np.concatenate((blue,green,red), axis = None)

            # Append to training data
            val_data[i] = img

    # Get predicted labels
    predicted_labels = svm.predict(val_data)

    # Get predicted probabilities
    probabilities = svm.predict_proba(val_data)
    probabilities = [elem [1] for elem in probabilities]

    # Delete validation data from memory
    del val_data

    # Compute and plot metrics
    precision, recall, specificity, f_score, bac = compute_and_plot_stats(true_labels, predicted_labels, probabilities, file_name)
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
    pickle.dump(svm, open('pretrained/' + file_name + '.p', 'wb' ))
    print("Done.")

def test(args) -> None:
    """
    Tests the SVC model

    Args:
        args: The arguments.
    """    

    global file_name, pca_comp

    file_name = args.file_name
    pca_comp = args.pca_comp

    # Paths to test data
    test_paths = glob(args.path + '*')

    if pca_comp is not None:
        print("Loading PCA matrix...")
        pca = load_pca_matrix(pca_comp)

    # Load SVM
    svm = pickle.load(open(file_name + '.p','rb'))

    # Matrix with validation samples
    true_labels = np.zeros(len(test_paths))
    test_data = np.zeros((len(test_paths),7500))

    if pca_comp is None:
        for i in tqdm(range(len(test_paths)), desc = 'Loading training data...'):

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
        
    else:
        for i in tqdm(range(len(test_paths)), desc = 'Loading training data with PCA...'):

            # Get path
            path = test_paths[i]

            # Get label
            true_labels[i] = int(path[-5])

            # Load image using OpenCV (not PIL, this is done this way to use PCA correctly)
            img = cv2.cvtColor( cv2.imread(path), cv2.COLOR_BGR2RGB )

            # Split image into RGB channels
            blue, green, red = cv2.split(img)

            # Normalize
            blue, green, red = blue/255, green/255, red/255

            # Project data to lower dimensions
            red = pca['red'].transform([ red.flatten() ])
            green = pca['green'].transform([ green.flatten() ])
            blue = pca['blue'].transform([ blue.flatten() ])

            # Reconstruct data
            red = pca['red'].inverse_transform(red)
            green = pca['green'].inverse_transform(green)
            blue = pca['blue'].inverse_transform(blue)

            # Concatenate channels
            img = np.concatenate((blue,green,red), axis = None)

            # Append to training data
            test_data[i] = img

    # Get predicted labels
    predicted_labels = svm.predict(test_data)

    # Get predicted probabilities
    probabilities = svm.predict_proba(test_data)
    probabilities = [elem [1] for elem in probabilities]

    # Compute and plot metrics
    precision, recall, specificity, f_score, bac = compute_and_plot_stats(true_labels, predicted_labels, probabilities, file_name)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Specificity:", specificity)
    print("F - Score:", f_score)
    print("Balanced Accuracy:", bac)

    # Confidence interval
    interval = interval95( bac / 100, len(test_paths))
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