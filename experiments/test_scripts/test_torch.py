# Import necessary libraries
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.VGG import VGGClassifier
from utils.loader import get_datasets
from models.BetterCNN import BetterCNN
# Define your datasets

def train_model(train_set, test_set, val_set, model):
    # Define your loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Set the number of epochs
    num_epochs = 10

    # Training loop
    for epoch in range(num_epochs):
        # Set the model to training mode
        model.train()

        # Iterate over the training set
        for inputs, labels in train_set:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Evaluate the model on the validation set
        model.eval()
        with torch.no_grad():
            total_correct = 0
            total_samples = 0
            for inputs, labels in val_set:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

            accuracy = total_correct / total_samples
            print(f"Epoch {epoch+1}: Validation Accuracy = {accuracy}")

    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for inputs, labels in test_set:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

        accuracy = total_correct / total_samples
        print(f"Test Accuracy = {accuracy}")

    

train1, test1, val1 = get_datasets("C:/Projekt/Vindr/Vindr")
model = BetterCNN(2)

train_model(train1, test1, val1, model)

train2, test2, val2 = get_datasets("C:/Projekt/rsna-bcd/split")
model = VGGClassifier(2)

train_model(train2, test2, val2, model)

train3, test3, val3 = get_datasets("C:/Projekt/DDSM/DDSM")
model = VGGClassifier(2)

train_model(train3, test3, val3, model)