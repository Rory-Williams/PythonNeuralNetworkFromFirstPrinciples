from math import exp
from math import sin
from math import log
from random import seed
from random import random
import numpy as np
import sys


def initialize_network(n_inputs, n_hid_lyr, n_hid_node, n_outputs):
	'''create network framework to store neurons'''
	network = list()
	for i in range(n_hid_lyr):
		if i==0:
			hidden_layer = [{'weights':[random() for j in range(n_inputs + 1)]} for k in range(n_hid_node[i])] #initialise the first number of weights for the hidden layer from the number of inputs plus 1 for bias
		else:
			hidden_layer = [{'weights':[random() for j in range(n_hid_node[i-1]+1)]} for k in range(n_hid_node[i])] #initialise the number of weights for each node on each hidden layer (with number of weights equal to the number of nodes on the previous hidden layer plus 1 for bias)
		network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hid_node[n_hid_lyr-1]+1)]} for i in range(n_outputs)] #initialise weights for each output from number of nodes of previous hidden layer (plus bias term)
	network.append(output_layer)
	return network


def activate(weights, inputs):
	'''Calculate neuron activation for an input. This is essentially just multiplying the inputs by weights and
	adding the bias (which is the weights[-1] bit)'''
	activation = weights[-1] #add the bias term to the activation summation
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i] #sum all the weights times inputs plus bias for the node to feed into the activation function
	return activation


def transfer(activation):
	'''Transfer neuron activation - set the activation function and apply to the sum from the 'activate' function'''
	return 1.0 / (1.0 + exp(-activation)) #sigmoid
# 	return np.tanh(activation) #tanh
# 	return np.maximum(0, activation) #relu
# 	return np.maximum(0.1 * activation, activation) #leaky_relu


# Forward propagate input to a network output - go down the layers calculating the output for each neuron, then setting that as the new input for the next neuron
# Network is dense, so each input is used with each neuron
def forward_propagate(network, row):
	'''Forward propagate input to a network output - go down the layers calculating the output for each neuron,
	then setting that as the new input for the next neuron Network is dense, so each input is used with each neuron'''
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation) # feed the (weight*input) +bias sum throughh the activation function and set as the neuron output ready to feed to the next level of neurons
			new_inputs.append(neuron['output']) # set inputs for next layer as outputs of this layer
		inputs = new_inputs
	return inputs


#
def transfer_derivative(output):
	'''Calculate the derivative of the neurons activation function to be used in backward propagation'''
	return output * (1.0 - output)  # sigmoid
# 	return 1 - np.square(output) #tanh
# 	if output/abs(output)>0:
# 		return 1 #relu & L_relu
# 	else:
# 		return 0 #relu
# 		return 0.1 #L_relu


def backward_propagate_error(network, expected):
	'''Backpropagate error and store each error for each neuron in its respective row in the network array
	The error for each neuron is found working back from the output layer For the hidden layers, the error is found from the
	upstream layers error multiplied by the neurons weight to find its contribution to the error Errors are finally
	converted to 'deltas' for each neuron by multiplying by the transfer/activation function derivative to weight how
	much the neuron needs to change'''
	for i in reversed(range(len(network))):  # work back from output layer
		layer = network[i]
		errors = list()
		if i != len(network) - 1:  # for all layers except output
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				# use the expected value which is the correct value for the output, found from the dataset given to the training function by the user
				errors.append(expected[j] - neuron['output'])  # MSE loss function derivative
		# 				errors.append( ( (expected[j]/neuron['output']) - ((1-expected[j])/(1-neuron['output'])) ) ) #Cross-entropy loss function derivative
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


def update_weights(network, row, l_rate, reg_mode, reg_param):
	'''Update network weights with error'''
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		# Update all weights for each neuron using the backpropagated delta values
		# First update the weights, then the bias
		# Apply damping regularisation terms as needed
		for neuron in network[i]:
			if reg_mode == 1:  # L1 regulation
				for j in range(len(inputs)):
					neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j] + l_rate * reg_param * (
								neuron['delta'] / abs(neuron['delta']))
				neuron['weights'][-1] += l_rate * neuron['delta'] + l_rate * reg_param * (
							neuron['delta'] / abs(neuron['delta']))
			if reg_mode == 2:  # L2 regularisation
				for j in range(len(inputs)):
					neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j] - l_rate * reg_param * \
											neuron['weights'][j]
				neuron['weights'][-1] += l_rate * neuron['delta'] - l_rate * reg_param * neuron['weights'][-1]
			else:  # No regularisation
				for j in range(len(inputs)):
					neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
				neuron['weights'][-1] += l_rate * neuron['delta']


def train_network(network, train, l_rate, n_epoch, n_outputs, reg_mode, reg_param):
	'''Train a network for a fixed number of epochs
	First forward propagate the network to get the outputs for each neuron next backpropagate using these outputs to find the error for each neuron
	Then update the weights of each neuron according to their error, the loss functions, learning rate, and derivative of the activation function
	Print out details of the total error of the network as it learns. Print out per run through all test values.'''
	sum_error_history = []
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			for i in range(n_outputs):
				expected[i] = row[i - n_outputs]
			sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])  # MSE
			# 			sum_error += sum([-( expected[i]*log(outputs[i])+(1-expected[i])*log(1-outputs[i])) for i in range(len(expected))]) #Cross-Entropy
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate, reg_mode, reg_param)
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
		sum_error_history.append(sum_error)
	return sum_error_history, sum_error



def predict(network, row):
	'''test new data set with trained network'''
	outputs = forward_propagate(network, row)
	return outputs


def complex_inputs(x1_input, x2_input, x12_input, x22_input, x_12_input, sin_x1_input, sin_x2_input, dataset,
				   n_outputs):
	'''Just adds aditional input columns after x1, x2 if functions of those inputs are called for
	First checks that the input is a nested list, else changes the index terms to be appropriate for a single list'''
	if isinstance(dataset, (np.ndarray)):
		dataset_width_og = len(dataset[0])
		dataset_width = len(dataset[0])
		dataset_len = len(dataset)

		if x12_input == 1:
			dataset_width += 1
			TempDataset = np.zeros((dataset_len, (dataset_width)))
			for i in range(len(dataset)):
				TempDataset[i] = np.insert(dataset[i], (dataset_width_og - n_outputs), dataset[i][0] ** 2)
			dataset = TempDataset

		if x22_input == 1:
			dataset_width += 1
			TempDataset = np.zeros((dataset_len, (dataset_width)))
			for i in range(len(dataset)):
				TempDataset[i] = np.insert(dataset[i], (dataset_width_og - n_outputs), dataset[i][1] ** 2)
			dataset = TempDataset

		if x_12_input == 1:
			dataset_width += 1
			TempDataset = np.zeros((dataset_len, (dataset_width)))
			for i in range(len(dataset)):
				TempDataset[i] = np.insert(dataset[i], (dataset_width_og - n_outputs), dataset[i][0] * dataset[i][1])
			dataset = TempDataset

		if sin_x1_input == 1:
			dataset_width += 1
			TempDataset = np.zeros((dataset_len, (dataset_width)))
			for i in range(len(dataset)):
				TempDataset[i] = np.insert(dataset[i], (dataset_width_og - n_outputs), sin(dataset[i][0]))
			dataset = TempDataset

		if sin_x2_input == 1:
			dataset_width += 1
			TempDataset = np.zeros((dataset_len, (dataset_width)))
			for i in range(len(dataset)):
				TempDataset[i] = np.insert(dataset[i], (dataset_width_og - n_outputs), sin(dataset[i][0]))
			dataset = TempDataset

		if x1_input != 1:
			dataset_width -= 1
			TempDataset = np.zeros((dataset_len, (dataset_width)))
			for i in range(len(dataset)):
				TempDataset[i] = np.delete(dataset[i], 0)
			dataset = TempDataset

		if x2_input != 1:
			dataset_width -= 1
			if x1_input != 1:
				x2_idx = 0
			else:
				x2_idx = 1
			TempDataset = np.zeros((dataset_len, (dataset_width)))
			for i in range(len(dataset)):
				TempDataset[i] = np.delete(dataset[i], x2_idx)
			dataset = TempDataset

	# For data sets of only a single row (and for graphing)
	else:
		dataset_width_og = len(dataset)
		dataset_width = len(dataset)

		if x12_input == 1:
			dataset_width += 1
			TempDataset = np.zeros(dataset_width)
			TempDataset = np.insert(dataset, (dataset_width_og - n_outputs), dataset[0] ** 2)
			dataset = TempDataset

		if x22_input == 1:
			dataset_width += 1
			TempDataset = np.zeros(dataset_width)
			TempDataset = np.insert(dataset, (dataset_width_og - n_outputs), dataset[1] ** 2)
			dataset = TempDataset

		if x_12_input == 1:
			dataset_width += 1
			TempDataset = np.zeros(dataset_width)
			TempDataset = np.insert(dataset, (dataset_width_og - n_outputs), dataset[0] * dataset[1])
			dataset = TempDataset

		if sin_x1_input == 1:
			dataset_width += 1
			TempDataset = np.zeros(dataset_width)
			TempDataset = np.insert(dataset, (dataset_width_og - n_outputs), sin(dataset[0]))
			dataset = TempDataset

		if sin_x2_input == 1:
			dataset_width += 1
			TempDataset = np.zeros(dataset_width)
			TempDataset = np.insert(dataset, (dataset_width_og - n_outputs), sin(dataset[0]))
			dataset = TempDataset

		if x1_input != 1:
			dataset_width -= 1
			TempDataset = np.zeros(dataset_width)
			TempDataset = np.delete(dataset, 0)
			dataset = TempDataset

		if x2_input != 1:
			dataset_width -= 1
			if x1_input != 1:
				x2_idx = 0
			else:
				x2_idx = 1
			TempDataset = np.zeros(dataset_width)
			TempDataset = np.delete(dataset, x2_idx)
			dataset = TempDataset

	# print(dataset)
	return dataset
