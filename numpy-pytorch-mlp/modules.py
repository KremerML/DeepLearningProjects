import numpy as np
import math
import torch

class LinearModule(object):

    def __init__(self, in_features, out_features, input_layer=False):

        # Weights
        self.w = np.random.randn(out_features, in_features)*math.sqrt(2/in_features)    
        # Biases 
        self.b = np.zeros(out_features)    
        
        # Gradients
        self.grads = {'weight': np.zeros((out_features, in_features)),
                      'bias': np.zeros((1,in_features))}

        
    def forward(self, x):
        # remember the inputs
        
        self.inputs = x
        b = np.tile(self.b, (x.shape[0], 1))
        self.out = np.dot(x, np.transpose(self.w)) + b

        return self.out

    
    def backward(self, dout):
        # Gradient of weights and biases
        dout = np.array(dout)
        
        self.grads['weight'] = np.dot(np.transpose(dout), self.inputs)
        self.grads['bias'] = np.dot(np.ones(dout.shape[0]), dout)
        # Gradient of values
        self.dx = np.dot(dout, self.w)
        
        return self.dx

    
    def clear_cache(self):
        pass



class ELUModule(object):
    """
    ELU activation module.
    """
        
    def forward(self, x):
        inputs = np.array(x)
        greater_than_0 = np.asarray(x > 0)
        less_than_or_equal_0 = np.asarray(x <= 0)

        self.out = (greater_than_0 * inputs) + (less_than_or_equal_0 * (np.exp(inputs) - 1))
        self.inputs = inputs
        return self.out

    def backward(self, dout):
        greater_than_0 = np.asarray(self.inputs > 0)
        less_than_or_equal_0 = np.asarray(self.inputs <= 0)
        dh = greater_than_0 + less_than_or_equal_0 * np.exp(self.inputs)
        
        self.dx = np.array(dout) * dh
        return self.dx

    def clear_cache(self):
        pass



class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        # remember inputs
        self.inputs = x

        # Unnormalized values
        exp_val = np.exp(x - np.max(x, axis=1, keepdims=True))
        
        # Values normalized for each sample
        norm_val = exp_val / np.sum(exp_val, axis=1, keepdims=True)
        self.out = norm_val
        return self.out

    def backward(self, dout):
        # Array of Gradients
        self.der_inputs = np.empty_like(dout)
        
        for idx, (one_output, one_grad) in enumerate(zip(self.out, dout)):
            # flatten the input
            one_output = one_output.reshape(-1, 1)
            
            # Calculate Jacobian matrix
            jacobian = np.diagflat(one_output) - np.dot(one_output, one_output.T)
            
            # calculate point-wise gradient and add to array of gradients
            self.der_inputs[idx] = np.dot(jacobian, one_grad)
        
        return self.der_inputs

    def clear_cache(self):
        pass



class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        pred_class = np.array(x)
        R, C = pred_class.shape
        true_class = np.eye(C)[y] 
        loss = -np.sum(true_class * np.log(pred_class), axis = 1)
        out = 1/R * np.sum(loss)
        
        return out
    
     
    def backward(self, x, y):
        x = np.array(x)
        R, C = x.shape
        
        truth = np.eye(C)[y]
        dloss = -1/R * (truth / x)
        
        self.dx = x * (dloss - (dloss * x) @ np.ones((C, C)))

        return self.dx