from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import modules as mm
from copy import deepcopy


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP
        """
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        
        n_hidden.append(n_classes)
        
        self.input_layer = mm.LinearModule(n_inputs, n_hidden[0])

        self.layers = [[None, self.input_layer]]
        for i in range(len(n_hidden)):
                if i>0:
                    self.layers.append([mm.ELUModule(), 
                                               mm.LinearModule(n_hidden[i-1], n_hidden[i])])
        
        self.layers = np.array(self.layers)
        
        self.softmax = mm.SoftMaxModule()

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """
        out = x
        
        for i, (activation, layer) in enumerate(self.layers):
            if i == 0:
                out = layer.forward(out)
            else:
                out = activation.forward(out)
                out = layer.forward(out)
                
        out = self.softmax.forward(out)
            
        return out

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss

        """

        dout = self.softmax.backward(dout)
        
        for i, (layer, activation) in enumerate(np.flip(self.layers)):
            if i != len(self.layers)-1:
                dout = layer.backward(dout)
                dout = activation.backward(dout)
            else:
                dout = layer.backward(dout)
         
    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass from any module.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Iterate over modules and call the 'clear_cache' function.
        """
        pass
