from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

import torch


class SGD:
    def __init__(self, learning_rate = 1e-2):
        self.learning_rate = learning_rate

    def update_parameters(self, model):
        
        for layer in model.layers[:,1]:
            # print(layer.b.shape)
            # print(layer.grads['bias'].shape)
            layer.w += -self.learning_rate * layer.grads['weight']
            layer.b += -self.learning_rate * layer.grads['bias']
        
        # for activation,layer in model.layers:
        #     layer.w += -self.learning_rate * layer.grads['weight']
        #     layer.b += -self.learning_rate * layer.grads['bias']


def confusion_matrix(predictions, targets):
    """
    Computes the confusion matrix, i.e. the number of true positives, false positives, true negatives and false negatives.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      confusion_matrix: confusion matrix per class, 2D float array of size [n_classes, n_classes]
    """
    S, N = predictions.shape
    conf_mat = np.zeros((N, N))
    
    targets = np.array(targets)
    preditions = np.argmax(predictions, 1)

    for pred, target in zip(preditions, targets):
        conf_mat[pred][target] += 1
    
    return conf_mat

    # confusion_matrix = np.zeros((predictions.shape[1],) * 2)
    # pred_labels = predictions.argmax(1)
    
    # for pred, truth in zip(pred_labels, targets):
    #     confusion_matrix[truth, pred] += 1
    # return confusion_matrix

def confusion_matrix_to_metrics(confusion_matrix, beta=1.):
    """
    Converts a confusion matrix to accuracy, precision, recall and f1 scores.
    Args:
        confusion_matrix: 2D float array of size [n_classes, n_classes], the confusion matrix to convert
    Returns: a dictionary with the following keys:
        accuracy: scalar float, the accuracy of the confusion matrix
        precision: 1D float array of size [n_classes], the precision for each class
        recall: 1D float array of size [n_classes], the recall for each clas
        f1_beta: 1D float array of size [n_classes], the f1_beta scores for each class
    """
    accuracy = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
    precision = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1))
    recall = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=0))
    f1_beta = (1 + beta*2) * (precision * recall) / (beta*2 * precision + recall)
    
    metrics = {'accuracy': accuracy, 
               'precision': precision, 
               'recall': recall, 
               'f1_beta': f1_beta}
    return metrics


def evaluate_model(model, data_loader, num_classes=10):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
        metrics: A dictionary calculated using the conversion of the confusion matrix to metrics.
    """

    conf_mat = np.zeros([num_classes, num_classes])
    for image, label in tqdm(data_loader):
        image = image.flatten().reshape(image.shape[0], 3072)
        prediction = model.forward(image)
        conf_mat += confusion_matrix(prediction, label)
    
    metrics = confusion_matrix_to_metrics(conf_mat, beta=1.)
    
    return metrics
    
def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation. Between 0.0 and 1.0
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)
    
    # setting train, val, and test dataloaders
    trainloader = cifar10_loader['train']
    val_loader = cifar10_loader['validation']
    testloader = cifar10_loader['test']
    
    # Initializing MLP model with 3072 input size, n_hidden, and 10 classes
    model = MLP(3072, hidden_dims, 10)
    
    # Initializing loss and optimizer
    crossEntropy = CrossEntropyModule()
    optimizer = SGD(lr)
    
    # Keeping track of losses, validation accuracy
    losses = []
    val_accuracies = []
    
    # Keeping track of the best model
    top_acc = 0
    data_inputs, data_labels = next(iter(trainloader))
    new_shape = np.prod(data_inputs.shape[1:])
    n_classes = max(data_labels) + 1
    
    # Iterate over epochs
    for epoch in tqdm(range(epochs)):
        # Calculate loss per batch
        batch_loss = []
        # Iterate over batches
        for image, labels in trainloader:
            
            image = np.reshape(image, (batch_size, new_shape))
            prediction = model.forward(image)
            
            loss = crossEntropy.forward(prediction, labels)
            batch_loss.append(loss)
            
            # backward pass
            dloss = crossEntropy.backward(prediction, labels)
            model.backward(dloss)
            optimizer.update_parameters(model)
        
        losses.append(np.mean(batch_loss)) 
        
        # Get metrics of model per epoch
        metrics = evaluate_model(model, val_loader)
        
        accuracy = metrics['accuracy']
        val_accuracies.append(metrics['accuracy'])
        
        # save best model
        if (accuracy > top_acc):
            top_acc = accuracy
            best_model = deepcopy(model)
            top_ep = epoch+1

        test_metrics = evaluate_model(best_model, testloader)
        test_accuracy = test_metrics['accuracy']

        logging_dict = {
        'epoch': epochs,
        'top_epoch': top_ep,
        'accuracy': val_accuracies,
        'top_accuracy': top_acc,
        'losses': losses,
        'dataloader': cifar10_loader,
        'batch_size': batch_size
        }
        
    return model, val_accuracies, test_accuracy, logging_dict
 

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    train(**kwargs)
    