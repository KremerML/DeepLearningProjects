from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from collections import OrderedDict


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes, use_batch_norm=False):
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
          use_batch_norm: If True, add a Batch-Normalization layer in between
                          each Linear and ELU layer.
        """
        
        super(MLP, self).__init__()
        
        features = torch.cat((torch.tensor([n_inputs]), 
                              torch.tensor(n_hidden), 
                              torch.tensor([n_classes])),0)
        
        self.layers = nn.ModuleList()
        
        for i in range(1, len(features[:-1])):                
                linear = nn.Linear(features[i-1], features[i])
                nn.init.kaiming_normal_(linear.weight)
                self.layers.append(linear)
                
                if use_batch_norm == True:
                    self.layers.append(nn.BatchNorm1d(features[i]))
                    
                self.layers.append(nn.ELU())
            
        linear = nn.Linear(features[-2], features[-1])
        nn.init.kaiming_normal_(linear.weight)
            
        self.layers.append(linear)
        if (use_batch_norm == True):
            self.layers.append(nn.BatchNorm1d(features[-1]))
            
    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """
        out = x.view(x.shape[0], -1)
        for layer in self.layers:
            out = layer.forward(out)
            
        return out
 
    @property
    def device(self):
        """
        Returns the device on which the model is. Can be useful in some situations.
        """
        return next(self.parameters()).device
    
