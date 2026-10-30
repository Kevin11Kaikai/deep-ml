import numpy as np

def sigmoid(X):
	return 1 / (1+np.exp(-X))

def lr_forward(coefficients,X):
	# logistic regression forward
	return sigmoid(X @ coefficients)

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
	"""
	# logistic regression backward
	losses = []
	D = X.shape[-1]
	L = X.shape[0]
	coefficients = np.zeros(D+1)
	X = np.concatenate([np.ones((L,1)),X], axis=-1)
	for i in range(iterations):
		output = lr_forward(coefficients,X)
		# lr forward
		error = output - y
		gradient =  X.T @ error
		coefficients -= learning_rate * gradient
		# coeffients update
		loss = 0
		sample_loss = y * np.log(output) + (1-y) * np.log(1-output)
		loss = -np.sum(sample_loss, axis=-1)
		losses.append(loss)	
		# losses update	
	coefficients = np.round(coefficients, 4).tolist()
	losses = np.round(losses, 4).tolist()
		
	return coefficients, losses