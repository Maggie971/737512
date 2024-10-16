import numpy as np


class Softmax:
    def __init__(self):
        pass

    def forward(self, inputs):
        # Subtracting the max value from each input to prevent overflow
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalizing the values for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities


# Example usage
softmax = Softmax()

# Simulating raw output (logits) from a neural network for 3 samples and 4 classes
logits = np.array([[1.0, 2.0, 3.0, 6.0],
                   [2.0, 4.0, 6.0, 8.0],
                   [1.0, 0.5, 0.2, 0.1]])

# Calculating softmax probabilities
probabilities = softmax.forward(logits)

print("Softmax probabilities:")
print(probabilities)
