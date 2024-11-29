
import numpy as np

# Defining the dataset
features_x = np.array([
    [1, 3, 3, 4],
    [2, 3, 2, 1],
    [3, 4, 1, 5],
    [1, 1, 2, 3],
    [2, 1, 2, 4],
    [1, 1, 1, 5],
    [3, 2, 1, 2],
    [2, 3, 1, 4],
    [4, 1, 2, 3],
    [5, 3, 1, 1],
    [2, 3, 3, 4],
    [4, 2, 4, 5]
])

labels_y = np.array([
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1]
])

weights_w = np.array([
    [1, 2, 1, 2],
    [3, 2, 1, 3],
    [1, 1, 2, 3],
    [3, 2, 1, 2]
])

biases_b = np.array([-6, -9, -7, -8])

# Function to calculate scores based on features, weights, and biases
def score_calculation(features, weights, bias):
    scores = []
    for i in features:
        array_x = np.array(i)
        score_row = []
        for j, b in zip(weights, bias):
            array_w = np.array(j)
            score_without_bias = sum(array_x * array_w)
            score_with_bias = score_without_bias + b
            score_row.append(score_with_bias)
        scores.append(score_row)
    scores = np.array(scores)
    return scores

# Function to calculate the softmax of the scores
def softmax_function(score):
    softmax = []
    for i in score:
        exp_array = np.exp(i)
        sum_exp_array = np.sum(exp_array)
        softmax_row = exp_array / sum_exp_array
        softmax.append(softmax_row)
    softmax = np.array(softmax)
    return softmax

# Function to calculate the cross-entropy loss
def cross_entropy_function(softmax, labels):
    cross_entropy_row = []
    indiv_entropy = -1 * np.log(softmax) * labels
    for i in indiv_entropy:
        cross_entropy_row.append(sum(i))
    cross_entropy = np.array(cross_entropy_row)
    return cross_entropy

# Function to calculate the error between the predicted probabilities and the actual labels
def error_function(labels, softmax):
    return softmax - labels

# Function to calculate the gradient of the loss function with respect to the features
def gradient_function(features, error):
    gradient_row = []
    for col in range(error.shape[1]):
        error_col = error[:, col]
        error_col_vector = error_col[:, np.newaxis]
        grad_val = error_col_vector * features
        gradient_row.append(grad_val)
    gradient = np.array(gradient_row)
    return gradient

# Function to update the weights based on the gradient and learning rate
def weights_update_function(gradient, learning_rate, weights):
    updated_weights_row = []
    for i in range(len(weights)):
        gradient_matrix = gradient[i]
        upd_w = []
        for grad_row in gradient_matrix:
            upd_weight = weights[i] - grad_row * learning_rate
            upd_w.append(upd_weight)
        updated_weights_row.append(upd_w)
    updated_weights = np.array(updated_weights_row)
    return updated_weights

# Function to perform one iteration of logistic regression training
def logistic_trick(weights, bias, features, label, learning_rate):
    score_array = score_calculation(features, weights, bias)
    softmax_array = softmax_function(score_array)
    cross_entropy_array = cross_entropy_function(softmax_array, label)
    error_array = error_function(label, softmax_array)
    gradient_array = gradient_function(features, error_array)
    updated_weights = weights_update_function(gradient_array, learning_rate, weights)
    updated_biases = bias - learning_rate * np.sum(error_array, axis=0)
    return updated_weights, updated_biases

# Function to train the logistic model iteratively for a specified number of epochs
def train_logistic_model(weights, biases, features, labels, learning_rate, epochs):
    for epoch in range(epochs):
        weights, biases = logistic_trick(weights, biases, features, labels, learning_rate)
        print(f"Epoch {epoch + 1}/{epochs} - Updated Weights:\n{weights}\nUpdated Biases:\n{biases}\n")
    return weights, biases

# Example usage
updated_weights, updated_biases = train_logistic_model(weights_w, biases_b, features_x, labels_y, 0.5, 10)
print("Final Updated Weights:\n", updated_weights)
print("Final Updated Biases:\n", updated_biases)
# # Example usage
# updated_weights, updated_biases = logistic_trick(weights_w, biases_b, features_x, labels_y, 0.5)
# print("Updated Weights:\n", updated_weights)
# print("Updated Biases:\n", updated_biases)
