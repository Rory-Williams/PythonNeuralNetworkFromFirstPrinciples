# external imports
from math import exp
from math import sin
from math import log
from random import seed
from random import random
import matplotlib.pyplot as plt
import numpy as np
import sys

# local imports
from datasets import Quad_dataset, Circle_dataset
from NN_Functions import initialize_network, activate, transfer, forward_propagate, transfer_derivative, \
    backward_propagate_error, update_weights, train_network, predict, complex_inputs


# --------------Initialise variables and dataset--------------
# Call the dataset to train on (circle or quad available)
# dataset_og = Circle_dataset
dataset_og = Quad_dataset
# seed(1) #Used to fix the random numbers of the weights to analyse specific changes to the network without random setting intervention
n_inputs = 2  # number of input nodes (complex inputs only defined for 2 inputs)
n_outputs = 1  # number of outputs, should equal outputs set in training data, usually equal to number of categories
n_hid_lyr = 1  # number of hidden layers
n_hid_node = [6]  # number of nodes per hidden layer, must be length = number of hidden layers


# ----------------sanity check users dataset parameters--------------------
# Kick out if inputs and outputs not possible with dataset length
if (n_inputs + n_outputs) != len(dataset_og[0]):
    sys.exit('Too many inputs & outputs selected for dataset')

# Kick out if hid_lyr and hid_node length not same
if n_hid_lyr != len(n_hid_node):
    if len(n_hid_node) == 0:
        sys.exit('Number of hidden layer nodes not specified')
    temp_hid_node = []
    for i in range(n_hid_lyr):
        temp_hid_node[i] = n_hid_node[i]
    n_hid_node = temp_hid_node


# -----------Learning rate, regularisation and input parameter setup-------------
# Set learning rate and regularisation
l_rate = 0.3  # learning rate
n_epoch = 2000  # number of iterations through training data
reg_mode = 0  # 0 = no regularisation, 1 = l1 regularisation, 2 = l2 regularisation
reg_param = 0.0007  # regularisation parameter
sum_error_history = []  # initialise error parameter

# Set inputs - ONLY works with 1 or 2 inputs, dont ask for x2 complex inputs if only have a single input node!
# Havent added sanity check code as wish to move onto other things
x1_input = 1
x2_input = 1
x12_input = 1  # x1^2
x22_input = 1  # x2^2
x_12_input = 0  # x1*x2
sin_x1_input = 0  # sin(x1)
sin_x2_input = 0  # sin(x2)


# -------------------------------End of setup-------------------------------------
# Mod dataset for required inputs
dataset = complex_inputs(x1_input, x2_input, x12_input, x22_input, x_12_input, sin_x1_input, sin_x2_input, dataset_og,
                         n_outputs)
n_inputs = len(dataset[0]) - n_outputs

# Initialise and train network
network = initialize_network(n_inputs, n_hid_lyr, n_hid_node, n_outputs)
sum_error_history, sum_error = train_network(network, dataset, l_rate, n_epoch, n_outputs, reg_mode, reg_param)
print(' ')
print('No. of iterations: ' + "{}".format(len(sum_error_history)))
print(' ')

# Output final network values
for layer in network:
    print(layer)

# Plot convergence graph
graph_x = range(len(sum_error_history))
plt.plot(graph_x, sum_error_history)

# Plot scatter of training data
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
for row in dataset_og:
    if row[2] == 0:
        ax.scatter(row[0], row[1], color='b')
    else:
        ax.scatter(row[0], row[1], color='r')


# -------------------------------Plot mapped space-------------------------------
# Initialise graph space (with evenly spaced graph points)
x = np.linspace(-2, 2, 51)
y = np.linspace(-2, 2, 51)
xv, yv = np.meshgrid(x, y)

# Calculate predictions for graph space
res1, res2 = np.meshgrid(x, y)
# row = np.zeros(2)
for i in range(len(x)):
    for j in range(len(y)):
        row = [xv[i, j], yv[i, j]]
        row = complex_inputs(x1_input, x2_input, x12_input, x22_input, x_12_input, sin_x1_input, sin_x2_input, row,
                             0)  # edit graph inputs same as the model for the prediction
        prediction = predict(network, row)
        res1[i, j] = prediction[0]

    # Better definition of boundary edge
for i in range(len(x)):
    for j in range(len(y)):
        if res1[i, j] > 0.5:
            res1[i, j] = 1
        else:
            res1[i, j] = 0

# -------------------------------Plot contour and original points-------------------------------
fig = plt.figure()
h = fig.add_axes([0, 0, 1, 1])
h.contourf(xv, yv, res1)
for row in dataset_og:
    if row[-1] == 0:
        h.scatter(row[0], row[1], color='b')
    else:
        h.scatter(row[0], row[1], color='r')


plt.show()

