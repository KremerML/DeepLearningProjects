import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

from cifar100_utils import get_train_validation_set, get_test_set


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(num_classes=100):
    """
    Returns a pretrained ResNet18 on ImageNet with the last layer
    replaced by a linear layer with num_classes outputs.
    Args:
        num_classes: Number of classes for the final layer (for CIFAR100 by default 100)
    Returns:
        model: nn.Module object representing the model architecture.
    """
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = models.resnet18(weights='IMAGENET1K_V1')   # load resnet18 model
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features     # extract fc layers features
    model.fc = nn.Linear(num_features, num_classes) # (num_of_classes)
    torch.nn.init.zeros_(model.fc.bias)
    torch.nn.init.normal_(model.fc.weight, mean=0.0, std=0.01)

    return model


def train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device, augmentation_name=None):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation.
        device: Device to use.
        augmentation_name: Augmentation to use for training.
    Returns:
        model: Model that has performed best on the validation set.
    """
    # Load the datasets
    train, validation = get_train_validation_set(data_dir, validation_size=5000, 
                                                 augmentation_name=augmentation_name)
    
    train_loader = DataLoader(train, batch_size=batch_size, 
                                  shuffle=True, drop_last=True,collate_fn=None)
    
    val_loader = DataLoader(validation, batch_size=batch_size, 
                                       shuffle=False, drop_last=False,collate_fn=None)    

    # Initialize the optimizer (Adam) to train the last layer of the model.
    model = model.to(device)
    
    loss_module = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop with validation after each epoch. Save the best model.
#     epoch_loss = []
#     epoch_accuracy = []
    validation_loss = []
    validation_accuracy = []
    max_accuracy = 0
    best_params = model.state_dict()
    
    for epoch in tqdm(range(epochs)):
        batch_loss = 0 # running loss
        batch_TP = 0 # running True Positives
        
        val_loss = 0
        val_TP = 0
        
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            
            images = images.to(device)
            labels = labels.to(device)
            
            # forward
            optimizer.zero_grad()
            preds = model(images)
            loss = loss_module(preds, labels)
            
            # backward
            loss.backward()
            optimizer.step()
            
#             batch_loss += loss.item() * images.size(0)
#             batch_TP += torch.sum(preds == labels.data)
            
#         epoch_loss.append(batch_loss / len(train))
#         epoch_accuracy.append(batch_TP / len(train) * 100)
        
        val_accuracy = evaluate_model(model, val_loader, device)
        validation_accuracy.append(val_accuracy)
        
        if val_accuracy > max_accuracy:
            max_accuracy = val_accuracy
            best_params = model.state_dict()
    
    # Load the best model on val accuracy and return it.
    torch.save(model.state_dict(), checkpoint_name)
    
    best_model = model.to(device)
    best_model.load_state_dict(best_params)
    
    metrics = {'epochs': epochs,
               'batch_size': batch_size,
               'val_loss': validation_loss,
               'val_accuracy': validation_accuracy}

    # Remove metrics before final submission
    return best_model, metrics


def conf_matrix(predictions, targets):
    conf_mat = torch.zeros((predictions.shape[1], predictions.shape[1]))
    preditions = torch.argmax(predictions, 1)

    for pred, target in zip(preditions, targets):
        conf_mat[pred][target] += 1
    return conf_mat


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    """
    conf_mat = torch.zeros((100, 100))
    
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            pred = model.forward(images)
            conf_mat += conf_matrix(pred, labels)

    no_diag = conf_mat - torch.diag(torch.diag(conf_mat))
    total_sum = torch.sum(conf_mat)
    
    true_positive = torch.diag(conf_mat)
    false_positive = torch.sum(no_diag, axis = 1)
    false_negative = torch.sum(no_diag, axis = 0)
    true_negative = []

#     precision = torch.divide(true_positive, (true_positive + false_positive))
#     recall = torch.divide(true_positive, (true_positive + false_negative))
    accuracy = torch.sum(true_positive) / total_sum

    return accuracy

def main(lr, batch_size, epochs, data_dir, seed, augmentation_name):
    """
    Main function for training and testing the model.

    Args:
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        seed: Seed for reproducibility.
        augmentation_name: Name of the augmentation to use.
    """
    # Set the seed for reproducibility
    set_seed(seed)

    # Set the device to use for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = get_model(num_classes=100)

    # Get the augmentation to use
    

    # Train the model
    best_model, metrics = train_model(model=model, lr=lr, batch_size=batch_size, 
                                      epochs=epochs, data_dir=data_dir, 
                                      checkpoint_name='best_params', device=device, augmentation_name=None)

    test = get_test_set(data_dir)
    test_loader = DataLoader(test, batch_size=batch_size, 
                             shuffle=True, drop_last=True,collate_fn=None)
    test_accuracy = evaluate_model(best_model,test_loader, device)
    print(test_accuracy)
    # Evaluate the model on the test set
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Feel free to add more arguments or change the setup

    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')
    parser.add_argument('--epochs', default=30, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=123, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR100 dataset.')
    parser.add_argument('--augmentation_name', default=None, type=str,
                        help='Augmentation to use.')

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
