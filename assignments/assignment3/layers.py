import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Copy from previous assignment
    l2_reg_loss = reg_strength * np.sum(np.power(W, 2))
    grad=reg_strength*W*2
    return l2_reg_loss, grad



def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    if predictions.ndim == 1:
        predictions -= np.max(predictions)
        softmax = np.exp(predictions)/np.sum(np.exp(predictions))
    else:
        predictions -= np.max(predictions, axis=1).reshape(-1, 1)
        softmax = np.exp(predictions) / np.sum(np.exp(predictions), axis=1).reshape(-1, 1)
    return softmax


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    len_trg_ind = 1 if type(target_index) is int else len(target_index)
    N = probs.shape[0] if len(probs.shape) == 1 else probs.shape[1]

    trgt = np.zeros((len_trg_ind, N))
    trgt[range(len_trg_ind), np.ravel(target_index)] += 1
    cross_entropy = -np.sum(trgt*np.log(probs))

#     raise Exception("Not implemented!")
    return cross_entropy 


def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # TODO: Copy from the previous assignment
    
    preds = preds.copy()
    probs = softmax(preds)
    loss = cross_entropy_loss(probs, target_index)
    if np.ndim(probs) == 1:
        probs[np.ravel(target_index)] -= 1
    else:
        probs[range(len(target_index)), np.ravel(target_index)] -= 1
    dprediction = probs

    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO copy from the previous assignment
        self.X_mask = (X>0)
        return np.where(X>0, X, 0)

    def backward(self, d_out):
        # TODO copy from the previous assignment
        d_result = d_out*self.X_mask
        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO copy from the previous assignment
        self.X = X.copy()
        return X.dot(self.W.value) + self.B.value

    def backward(self, d_out):
        # TODO copy from the previous assignment
        self.W.grad = self.X.transpose().dot(d_out)
        E = np.ones(shape=(1, self.X.shape[0]))
        self.B.grad = E.dot(d_out)
        d_input = d_out.dot(self.W.value.transpose())

        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape

#         out_height = 0
#         out_width = 0
        out_height = height - self.filter_size + 2 * self.padding + 1
        out_width = width - self.filter_size + 2 * self.padding + 1
        
        self.X = np.pad(X, ((0,0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant', constant_values = 0)
        W = self.W.value.reshape(self.filter_size * self.filter_size * self.in_channels, self.out_channels)
        
        out_shape = (batch_size, out_height, out_width, self.out_channels)
        output = np.zeros(out_shape)
        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            for x in range(out_width):
                
                # TODO: Implement forward pass for specific location
                h_end, w_end = y + self.filter_size, x + self.filter_size
                
                I = self.X[:, y:h_end, x:w_end, :].reshape(batch_size, -1)
                output[:, y, x, :] = np.dot(I, W)
                
        return output + self.B.value


def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        # Try to avoid having any other loops here too
        # d_inp = np.zeros((batch_size, height - 2 * self.padding, width - 2 * self.padding, channels))
        d_inp = np.zeros(self.X.shape)
        window = np.zeros(self.X.shape)
        for y in range(out_height):
            for x in range(out_width):
                # d_cube is shape (batch_size, out_channel) => (batch_size, out_features)
                d_cube = d_out[:, y, x, :]
                # X_cube is shape (batch_size, filter_size, filter_size, channels)
                X_cube = self.X[:, y: y + self.filter_size, x: x + self.filter_size, :]
                # X_cube is shape (batch_size, filter_size * filter_size * channels) => (batch_size, in_features)
                #                                   0, 1, 2, 3
                X_cube = np.transpose(X_cube, axes=[0, 3, 1, 2]).reshape((batch_size, self.filter_size ** 2 * channels))
                # W_cube is shape (filter_size * filter_size * in_channels, out_shannel) => (in_features, out_features)
                W_cube = np.transpose(self.W.value, axes=[2, 0, 1, 3])
                W_cube = W_cube.reshape((self.filter_size ** 2 * self.in_channels, self.out_channels))
                # self.W.grad = self.X.transpose().dot(d_out)
                # E = np.ones(shape=(1, self.X.shape[0]))
                # self.B.grad = E.dot(d_out)
                # d_out.dot(self.W.value.transpose())
                # gradiants for dense layer reshaped to shape of W
                d_W_cube = (X_cube.transpose().dot(d_cube)).reshape(self.in_channels,
                                                                    self.filter_size,
                                                                    self.filter_size,
                                                                    self.out_channels)

                self.W.grad += np.transpose(d_W_cube, axes=[2, 1, 0, 3])
                E = np.ones(shape=(1, batch_size))
                self.B.grad += E.dot(d_cube).reshape((d_cube.shape[1]))

                # d_cube : (batch_size, out_features) dot W_cube.transpose: (out_features, in_features)
                # d_inp_xy is shape (batch_size, in_features)
                d_inp_xy = d_cube.dot(W_cube.transpose())
                d_inp_xy = d_inp_xy.reshape((batch_size, channels, self.filter_size, self.filter_size))
                #                                       0, 1, 2, 3
                d_inp_xy = np.transpose(d_inp_xy, axes=[0, 3, 2, 1])

                d_inp[:, y: y + self.filter_size, x: x + self.filter_size, :] += d_inp_xy
                window[:, y: y + self.filter_size, x: x + self.filter_size, :] += 1

        if self.padding:
            d_inp = d_inp[:, self.padding: -self.padding, self.padding: -self.padding, :]

        return d_inp

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        raise Exception("Not implemented!")

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement backward pass
        raise Exception("Not implemented!")

    def params(self):
        # No params!
        return {}
