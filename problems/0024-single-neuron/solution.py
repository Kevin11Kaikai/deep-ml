import math
import numpy as np
def sigmoid(X):
	output = 1 / (1+np.exp(-X))
	return output
def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Your code here
	features, labels, weights, bias = np.array(features), np.array(labels), np.array(weights), np.array(bias)
	# transfer lists to array
	output = features @ weights.T + bias
	# get probabilities
	probabilities = sigmoid(output)
	B = features.shape[0]
	probabilities = probabilities.reshape(B)
	# calculate mse
	mse = 0
	for i in range(B):
		mse += (probabilities[i]-labels[i])**2 / B
	return probabilities, mse