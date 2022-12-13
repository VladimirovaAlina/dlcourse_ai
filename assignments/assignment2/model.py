import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.model = [
            FullyConnectedLayer(n_input=n_input, n_output=hidden_layer_size),
            ReLULayer(),
            FullyConnectedLayer(n_input=hidden_layer_size, n_output=n_output)
        ]


    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        
        for param in self.params().values():
            param.grad = 0
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        X_ = X.copy()
        for layer in self.model:
            X_ = layer.forward(X_)
            
        loss, d_pred = softmax_with_cross_entropy(X_, y)
        
        d_out = d_pred.copy()
        for layer in reversed(self.model):
            d_out = layer.backward(d_out)

        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        
        if self.reg:
            for num, layer in enumerate(self.model):
                if isinstance(layer, FullyConnectedLayer):
                    l2_loss, d_reg = l2_regularization(layer.W.value, self.reg)
                    loss += l2_loss
                    layer.W.grad += d_reg

            # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!

        return loss
    

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        out = np.zeros(X.shape[0], np.int)
        out = self.model[2].forward(self.model[1].forward(self.model[0].forward(X)))
        pred = np.argmax(out, axis = 1)


        return pred

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params

   

        return result
