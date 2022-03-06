import sys
import time
import os
import math

from scipy.interpolate import make_interp_spline

import numpy as np
from numpy import linspace as linspace
import matplotlib.pyplot as plt

import torch
from torch import mean

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

TOTAL_BAR_LENGTH = 70
last_time = time.time()
begin_time = last_time

# Uncomment both lines if you want to see the output in the terminal
# _, term_width = os.popen('stty size', 'r').read().split()
# term_width = int(term_width)

# Comment this next linei f you want to see the output in the terminal
term_width = 100

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):

    # Just in case it takes a long time...
    days = seconds // (3600/24)
    seconds = seconds - days*3600*24

    hours = seconds // 3600
    seconds = seconds - hours*3600

    minutes = seconds // 60
    seconds = seconds - minutes*60

    real_seconds = int(seconds)
    seconds = seconds - real_seconds

    millis = int(seconds*1000)

    f = ''
    # Maximum units to show is 2
    max_units = 1
    if days > 0:
        f += str(days) + 'D'
        max_units += 1

    if hours > 0 and max_units <= 2:
        f += str(hours) + 'h'
        max_units += 1
    if minutes > 0 and max_units <= 2:
        f += str(minutes) + 'm'
        max_units += 1
    if real_seconds > 0 and max_units <= 2:
        f += str(real_seconds) + 's'
        max_units += 1
    if millis > 0 and max_units <= 2:
        f += str(millis) + 'ms'
        max_units += 1

    # If it took less than 1ms
    if f == '':
        f = '0ms'

    return f

def interval95(acc,data):
    """
    Calculates accuracy confidence interval at 95%
    Args:
        acc: Accuracy (in %)
        data: Number of samples used for testing

    """
    
    bound = 1.96 * math.sqrt((acc*(1-acc)) / data)

    return [acc - bound, acc + bound]


def plot(x_axis, y_axis, x_label,y_label, name = "Plot"):
    """
    Creates plot using pyplot
    Args:
        x_axis = List with values for the x axis
        y_axis = List with values for the y axis (must contain tuples like: ([values], label))
        x_label = Label for the x axis
        y_label = Label for the y axis
        name: Name of the file where the plot will be saved

    """
    new_x_axis = linspace(min(x_axis), max(x_axis), 200)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    for y in y_axis:
        
        # There must be at least 4 values in x_axis and in y[0] for this to work
        spl = make_interp_spline(x_axis, y[0], k=3)
        
        y_smooth = spl(new_x_axis)

        plt.plot(new_x_axis, y_smooth, label= y[1])

    plt.legend()
    plt.savefig(name)
    plt.clf()

    # If saved correctly, return True
    return True


def get_mean_and_std(dataloader):
    """
    Computes mean and standard deviation of a dataset given its loader
    Args:
        dataloader: Dataset loader (could be any)
    """


    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in dataloader:
        # Computes mean over batch, height and width, not over the channels
        channels_sum += mean(data, dim=[0,2,3])
        channels_squared_sum += mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    data_mean = channels_sum / num_batches

    # Var[X] = E[X**2] - E[X] ** 2
    # Standard deviation
    std = (channels_squared_sum / num_batches  - data_mean**2) ** 0.5

    return data_mean, std
    
def count_parameters(model):
    """
    Returns the amount of trainable parameters in a model
    Args:
        model: NN model
    """

    return sum(param.numel() for param in model.parameters() if param.requires_grad)

    
def plot_confusion_matrix(model, dataloader,device):

    with torch.no_grad():
        y_pred = []
        y_true = []

        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            # Get predicted output
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            # Accumulate predictions
            y_pred.extend(predicted.tolist()) 
            
            # Accumulate correct labels
            labels = labels.data
            y_true.extend(labels.tolist()) 
        


        
        # Build confusion matrix
        cf_matrix = confusion_matrix(y_pred, y_true)

        # Information that will appear in each cell
        group_names = ['True Neg','False Pos','False Neg','True Pos']
        group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]

        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
        labels = labels[2:] + labels[:2]
        
        labels = np.asarray(labels).reshape(2,2)

        ax = sn.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues_r',cbar = False)
        ax.set_title(model.name + " confussion matrix")
        ax.set_xlabel('Predicted class')
        ax.set_ylabel('Actual class ')

        # Axis labels
        ax.xaxis.set_ticklabels(['Benign','Malignant'])
        ax.yaxis.set_ticklabels(['Malignant','Benign'])
        ax.set_ylim([0,2])
        fig = ax.get_figure()
        fig.savefig('Confussion_Matrix_' + model.name + '.png',dpi=400)

    
    return

