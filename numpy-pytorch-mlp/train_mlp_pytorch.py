from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from copy import deepcopy
from tqdm.auto import tqdm
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn
import torch.optim as optim


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

    conf_mat = torch.zeros((predictions.shape[1], predictions.shape[1]))
    preditions = torch.argmax(predictions, 1)

    for pred, target in zip(preditions, targets):
        conf_mat[pred][target] += 1
        
    return conf_mat


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
    metrics = {}
    
    no_diag = confusion_matrix - torch.diag(torch.diag(confusion_matrix))
    total_sum = torch.sum(confusion_matrix)
    
    true_positive = torch.diag(confusion_matrix)
    false_positive = torch.sum(no_diag, axis = 1)
    false_negative = torch.sum(no_diag, axis = 0)
    true_negative = []

    precision = torch.divide(true_positive, (true_positive + false_positive))
    recall = torch.divide(true_positive, (true_positive + false_negative))
    accuracy = torch.sum(true_positive) / total_sum
    f1_beta = (1 + beta**2) * (precision * recall)/(beta**2 * precision + recall)
    
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['accuracy'] = accuracy
    metrics['f1_beta'] = f1_beta
    
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    conf_mat = torch.zeros((num_classes, num_classes))
    
    model.eval()
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        pred = model.forward(images)
        conf_mat += confusion_matrix(pred, labels)

    metrics = confusion_matrix_to_metrics(conf_mat)

    return metrics


def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation.
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Set default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=False)

    model = MLP(3072, hidden_dims, 10, use_batch_norm).to(device)
    
    crossEntropy = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    losses = []
    val_accuracies = []
    val_f1_scores = []
    max_accuracy = 0
    
    for epoch in range(epochs):
        
        batch_loss = []
        
        model.train()
        for images, labels in tqdm(cifar10_loader['train']):
            
            images = images.to(device)
            labels = labels.to(device)
            
            # forward
            preds = model(images)
            loss = crossEntropy(preds, labels)
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_loss.append(loss)
        
        losses.append(torch.mean(torch.tensor(batch_loss)))
        
        metrics = evaluate_model(model, cifar10_loader['validation'])
        val_accuracies.append(metrics['accuracy'])
        val_f1_scores.append(metrics['f1_beta'])
        
        if (metrics['accuracy'] > max_accuracy):
            max_accuracy = metrics['accuracy']
            best_params = model.state_dict()

    best_model = MLP(3072, hidden_dims, 10, use_batch_norm).to(device)
    best_model.load_state_dict(best_params)
    
    test_metrics = evaluate_model(best_model, cifar10_loader['test'])
    test_accuracy = test_metrics['accuracy']
    
    logging_dict = {'epoch': epochs,
                    'accuracy': val_accuracies,
                    'max_accuracy': max_accuracy,
                    'f1_beta': val_f1_scores,
                    'losses': losses,
                    'data_loader': cifar10_loader,
                    'batch_size': batch_size}

    return model, val_accuracies, test_accuracy, logging_dict
 
if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='Use this option to add Batch Normalization layers to the MLP.')
    
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
    