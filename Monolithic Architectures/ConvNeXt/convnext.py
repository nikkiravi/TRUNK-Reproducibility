# ----------------------------------------------------
# Name: metrics.py
# Purpose: Script to train and test convnext
# Author: Nikita Ravi
# Created: February 24, 2024
# ----------------------------------------------------

# Import necessary libraries
import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
from metrics import *


# Global Variables
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device is on {device}")
device = torch.device(device)

loss_function = nn.CrossEntropyLoss() # loss function to compare target with prediction

@run_time
def train(model, trainloader, epochs, learning_rate, betas, weight_decay, save_to_path=None):
    """
    Train the model on the chosen dataset

    Parameters
    ----------
    model: torchvision.models
        the chosen model by the user

    trainloader: torch.utils.data.DataLoader
        iterable dataset 

    epochs: int
        number of epochs to train over

    learning_rate: float
        learning rate for optimizer

    weight_decay: float
        weight decay for optimizer

    save_to_path: str [Optional]
        path to save the model

    Return
    ------
    loss_per_iteration: list
        list of training loss over all the epochs and batches
    """

    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
    loss_per_iteration = []

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if((batch_idx + 1) % 100 == 0):
                print("[epoch: %d, batch: %5d] loss: %.3f" % (epoch, batch_idx + 1, running_loss / 100))
                loss_per_iteration.append(running_loss / 100)
                running_loss = 0.0

    if(save_to_path):
        torch.save(model.state_dict(), save_to_path)

    return loss_per_iteration

@run_time
def test(model, path_to_network, testloader, num_classes):
    """
    Conduct validation on the model for the chosen dataset

    Parameters
    ----------
    model: torchvision.models
        the chosen model by the user

    path_to_network: str
        path where model is saved

    testloader: torch.utils.data.DataLoader
        iterable dataset 

    num_classes: int
        number of classes

    Return
    ------
    confusion_matrix: np.array
        the confusion matrix for the dataset using this trained model

    accuracy: float
        the accuracy of the model
    """

    model.load_state_dict(torch.load(path_to_network))
    model = model.to(device)
    model.eval()
    confusion_matrix = np.zeros((num_classes, num_classes))

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)
            for label, prediction in zip(labels, predicted):
                confusion_matrix[label][prediction] += 1
            
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    return confusion_matrix, accuracy

def get_model(dataset, num_classes):
    """
    Get the Pre-Trained ConvNeXt Model

    Parameters
    ----------
    num_classes: int
        number of classes in the dataset

    dataset: str
        the dataset chosen by the user 

    Return
    ------
    convnext: torchvision.models
        pre-trained convnext model
        
    """

    convnext = models.convnext_base(pretrained=True)
    if(dataset == "emnist"):
        first_layer_channels = convnext.features[0].out_channels
        convnext.features[0] = nn.Conv2d(in_channels=1, out_channels=first_layer_channels, kernel_size=(4, 4), stride=(4, 4)) # ConvNeXt expects a 3-channel RGB image but EMNIST is a single channel grayscale image

    input_features = convnext.classifier[2].in_features
    convnext.classifier[2] = torch.nn.Linear(input_features, num_classes) # convnext is originally trained on imageNet's 1000 classes but cifar10 only has 10 classes
    return convnext

def get_hyperparameters(dataset):
    """
    Get the hyperparameters based on the dataset

    Parameters
    ----------
    dataset: str
        the dataset chosen by the user

    Return
    ------
    epochs: int
        number of epochs to train over

    learning_rate: float
        learning rate for optimizer

    weight_decay: float
        weight decay for optimizer
    """

    if(dataset == "cifar10"):
        epochs = 10
        learning_rate = 1e-4
        betas = (0.9, 0.99)
        weight_decay = 5e-4

    elif(dataset == "emnist"):
        epochs = 10
        learning_rate = 1e-4
        betas = (0.9, 0.99)
        weight_decay = 5e-4 

    elif(dataset == "svhn"):
        epochs = 10
        learning_rate = 0.5e-5
        betas = (0.9, 0.99)
        weight_decay = 1e-4

    return epochs, learning_rate, betas, weight_decay

def convnext(trainloader, testloader, dataset):
    """
    create a convnext model and conduct training and testing on the chosen dataset

    Parameters
    ----------
    trainloader: torch.utils.data.DataLoader
        the iterable training dataset

    testloader: torch.utils.data.DataLoader
        the iterable testing dataset

    dataset: str
        the dataset chosen by the user 
    """

    path_to_model_weights = f"ConvNeXt/convnext_weights_{dataset}.pt"
    image_size = (1,) + tuple(trainloader.dataset[0][0].shape)
    if(dataset == "svhn"):
        class_list = list(set(testloader.dataset.labels))
    else:
        class_list = testloader.dataset.classes
    num_classes = len(class_list)

    epochs, learning_rate, betas, weight_decay = get_hyperparameters(dataset)
    convnext = get_model(dataset, num_classes)

    training_loss = train(model=convnext, trainloader=trainloader, epochs=epochs, learning_rate=learning_rate, betas=betas, weight_decay=weight_decay, save_to_path=path_to_model_weights)
    plot_losses(loss=training_loss, epochs=epochs, dataset=dataset, path="ConvNeXt/")

    confusion_matrix, accuracy = test(model=convnext, path_to_network=path_to_model_weights, testloader=testloader, num_classes=num_classes)
    print(f"Validation Accuracy for ConvNeXt-Base: {accuracy}")
    display_confusion_matrix(conf=confusion_matrix, class_list=class_list, accuracy=accuracy, dataset=dataset, path="ConvNeXt/")

    # Print the size of the model
    print_size_of_model(path_to_model=path_to_model_weights, label="ConvNeXt-Base")

    # Print FLOPs and Number of Parameters
    num_operations(model=convnext.to("cpu"), input_size=image_size, label="ConvNeXt-Base")
