import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    
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
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        
        self.X_mask = None
        

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.X_mask = (X>0)
        
#         raise Exception("Not implemented!")
        
        return np.where(X>0, X, 0)
    
    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        d_result = d_out*self.X_mask
#         print(self.X_mask)
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X.copy()
        return X.dot(self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        self.W.grad = self.X.transpose().dot(d_out)
        E = np.ones(shape=(1, self.X.shape[0]))
        self.B.grad = E.dot(d_out)
        
        d_input = d_out.dot(self.W.value.transpose())

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}

    
    
    
    
    
    
