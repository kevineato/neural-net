import math
import matplotlib.pyplot as plt
import numpy as np

class NeuralNetClf(object):
    """A classifier for learning models using variable number of hidden layers
    and neurons.

    Attributes:
        _n_hlayers: Number of hidden layers.
        _n_features: Number of features for input.
        _same_n_hidden: Whether each hidden layer contains same number of nodes.
        _act_fn: The activation function to use for each of the hidden layers.
        _n_classes: The number of classes for output.
        _learning_rate: Learning rate used for optimization/gradient descent.
        weights: The weights of the network as a list of matrices (synapses)
                 between layers.
        biases: The biases of the network as a list of vectors between layers.
    """
    def __init__(self, n_hlayers, n_features, n_hidden=None, activation='sigmoid',
                 n_classes=2, learning_rate=0.001):
        """Initialize variable hidden layer neural network.

        Args:
            n_hlayers: Number of hidden layers in network.
            n_features: Size of input layer.
            n_hidden: Number of nodes in hidden layers. May be a single value
                      or list of values of shape (1, n_hlayers).
            activation: Activation function to use in hidden layers. May be one
                        of sigmoid, relu, or tanh.
            n_classes: Number of classes to predict for output layer.
            learning_rate: Learning rate for optimization of network.
        """
        self._n_hlayers = n_hlayers
        self._n_features = n_features

        self._same_n_hidden = False

        # If n_hidden None, use mean of n_features and n_classes
        if n_hidden == None:
            self._n_hidden = [math.floor((n_features + n_classes) / 2)]
        else:
            if isinstance(n_hidden, int):
                self._n_hidden = [n_hidden]
            else:
                self._n_hidden = n_hidden

        # Default to sigmoid if activation is not an available activation function
        if activation == 'relu':
            self._act_fn = self._relu
        elif activation == 'tanh':
            self._act_fn = self._tanh
        else:
            self._act_fn = self._sigmoid

        self._n_classes = n_classes
        self._learning_rate = learning_rate

        self.weights = []
        self.biases = []

        # Check correct size/usage of nodes in hidden layers
        try:
            assert len(self._n_hidden) == self._n_hlayers
        except AssertionError:
            try:
                assert len(self._n_hidden) == 1
                self._same_n_hidden = True
            except AssertionError:
                raise ValueError("Length of n_hidden must equal 1 or n_hlayers")

        # Initialize all weights and biases from random normal distribution
        for i in range(self._n_hlayers + 1):
            if self._same_n_hidden:
                if i == 0:
                    self.weights.append(np.random.randn(self._n_features,
                                                        self._n_hidden[0]))
                    self.biases.append(np.random.randn(self._n_hidden[0]))
                elif i == self._n_hlayers:
                    self.weights.append(np.random.randn(self._n_hidden[0],
                                                        self._n_classes))
                    self.biases.append(np.random.randn(self._n_classes))
                else:
                    self.weights.append(np.random.randn(self._n_hidden[0],
                                                        self._n_hidden[0]))
                    self.biases.append(np.random.randn(self._n_hidden[0]))
            else:
                if i == 0:
                    self.weights.append(np.random.randn(self._n_features,
                                                        self._n_hidden[i]))
                    self.biases.append(np.random.randn(self._n_hidden[i]))
                elif i == self._n_hlayers:
                    self.weights.append(np.random.randn(self._n_hidden[i - 1],
                                                        self._n_classes))
                    self.biases.append(np.random.randn(self._n_classes))
                else:
                    self.weights.append(np.random.randn(self._n_hidden[i - 1],
                                                        self._n_hidden[i]))
                    self.biases.append(np.random.randn(self._n_hidden[i]))

    def _relu(self, x, derivative=False):
        """Relu activation function for forward hidden layer activation.

        Args:
            x: Output from previous layer calculated with weights and biases.
            derivative: Whether to calculate activations derivative with respect
                        to x for back propagation.
        """
        # Avoid overflow in precision of floating point ops
        np.clip(x, np.finfo(x.dtype).min, np.finfo(x.dtype).max, out=x)
        if derivative:
            x[x <= 0.01] = 0.01
            x[x > 0.01] = 1
            return x

        return np.maximum(0.01, x)

    def _sigmoid(self, x, derivative=False):
        """Sigmoid activation function for forward hidden layer activation.

        Args:
            x: Output from previous layer calculated with weights and biases.
            derivative: Whether to calculate activations derivative with respect
                        to x for back propagation.
        """
        # Avoid overflow in precision of floating point ops
        np.clip(x, np.finfo(x.dtype).min, np.finfo(x.dtype).max, out=x)
        if derivative:
            return x * (1 - x)

        return 1 / (1 + np.exp(-x))

    def _softmax(self, x):
        """Softmax to produce output layer probabilities for class prediction
        """
        # Avoid overflow in precision of floating point ops
        np.clip(x, np.finfo(x.dtype).min, np.finfo(x.dtype).max, out=x)
        expx = np.exp(x)
        return expx / np.sum(expx, axis=1, keepdims=True)

    def _tanh(self, x, derivative=False):
        """Tanh activation function for forward hidden layer activation.

        Args:
            x: Output from previous layer calculated with weights and biases.
            derivative: Whether to calculate activations derivative with respect
                        to x for back propagation.
        """
        # Avoid overflow in precision of floating point ops
        np.clip(x, np.finfo(x.dtype).min, np.finfo(x.dtype).max, out=x)
        if derivative:
            return 1 - (x * x)

        # tanh = 2 * sigmoid(2x) - 1
        return 2 * self._sigmoid(2 * x) - 1

    def feed_forward(self, X):
        """Feeds input forward through the network.

        Args:
            X: The input to the network of shape (None, self._n_features)

        Returns:
            The resultant layers from this forward feed and their values at the
            nodes of the network as an array.
        """
        layers = []
        for i in range(self._n_hlayers + 1):
            if i == 0:
                curr_layer = self._act_fn(np.dot(X, self.weights[0]) +
                                          self.biases[0])
            elif i == self._n_hlayers:
                curr_layer = self._softmax(np.dot(curr_layer, self.weights[i]) +
                                           self.biases[i])
            else:
                curr_layer = self._act_fn(np.dot(curr_layer, self.weights[i]) +
                                          self.biases[i])
            layers.append(curr_layer)

        return layers

    def back_prop(self, X, y, layers):
        """Back propagation of the error and gradients so network may learn.

        Args:
            X: The original input data to the network of shape (None, self._n_features).
            y: The correct labels/classes for the input data X.
            layers: Layers returned from feed_forward call before this call.
        """
        # Always use softmax so final layer delta is always same
        ly_delta = layers[-1] - y
        curr_delta = ly_delta
        deltas = []
        # Go through network backwards and compute each error and delta by
        # propagating the previous layers error and delta
        for i in range(self._n_hlayers, 0, -1):
            deltas.append(curr_delta)
            error = np.dot(curr_delta, self.weights[i].T)
            curr_delta = error * self._act_fn(layers[i - 1], derivative=True)

        # Append first hidden layers delta
        deltas.append(curr_delta)

        # Update first synapse weights
        self.weights[0] -= self._learning_rate * np.dot(X.T, deltas[-1])
        self.biases[0] -= self._learning_rate * np.sum(deltas[-1])
        # Go through network updating all weights and biases with deltas and transpose layers
        for i in range(1, len(deltas)):
            self.weights[i] -= self._learning_rate * np.dot(layers[i - 1].T, deltas[-i - 1])
            self.biases[i] -= self._learning_rate * np.sum(deltas[-i - 1], axis=0)

    def train(self, X, y, num_epochs=10000, log_rate=None, show_plt=False, title=None):
        """Trains the network, teaching it to classify tuples.

        Args:
            X: Training data to learn from.
            y: The class/labels for each of the respective training data.
            num_epochs: Number of epochs to run training for where one epoch
                        is one pass of data forward and propagated backwards
                        through the network.
            log_rate: Rate to report current value of loss function which is
                      the softmax cross entropy.
            show_plt: Whether to display a plot of the loss over num_epochs
                      after training is finished.
            title: A title for the plot if plot is specified.
        """
        losses = []
        for epoch in range(num_epochs):
            # Feed forward
            layers = self.feed_forward(X)
            ly = layers[-1]

            # Back-propagation
            self.back_prop(X, y, layers)

            if log_rate != None and epoch % log_rate == 0:
                # Softmax cross entropy loss of predicted output and actual class
                loss = np.mean(-np.sum(y * np.log(ly), axis=1, keepdims=True))
                losses.append(loss)
                print("Loss value at step {}: {:.4f}".format(epoch, loss))

        if log_rate != None and show_plt:
            plt.plot(losses, label='Loss')
            if title is not None:
                plt.title(title)
            plt.xlabel('Num Epochs')
            plt.ylabel('Cross-entropy loss')
            plt.legend()
            plt.show()

    def predict(self, X):
        """Returns a prediction of class of given data.

        Args:
            X: The data on which to predict the classes.

        Returns:
            An array of shape (X.shape[0], _n_classes). Where the columns are
            a probability distribution for the given tuple in X belonging to the
            class at that given column index.
        """
        return self.feed_forward(X)[-1]

    def eval(self, X_test, y_test):
        """Evaluates the accuracy on test data after a network has been trained.

        Args:
            X_test: The test data to make predictions on.
            y_test: The correct classes/labels for the X_test data.

        Returns:
            The accuracy and count of correct tuple predictions on the test data,
            where the accuracy is a percentage, i.e. the number of correctly
            predicted tuples divided by the total number of tuples.
        """
        count = 0
        for i in range(X_test.shape[0]):
            if np.all(np.argmax(self.predict(X_test), axis=1)[i] == y_test[i]):
                count += 1

        return count / X_test.shape[0], count
