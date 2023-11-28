
import torch.nn as nn
from torch import flatten

class TorchCNN(nn.Module):
    def __init__(self):
        super(TorchCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8,
			kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		# initialize second set of CONV => RELU => POOL layers
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16,
			kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		# initialize first (and only) set of FC => RELU layers
        self.fc1 = nn.Linear(in_features=125*125*16, out_features=500)
        self.relu3 = nn.ReLU()
		# initialize our softmax classifier
        self.fc2 = nn.Linear(in_features=500, out_features=2)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
		# pass the input through our first set of CONV => RELU =>
		# POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc2(x)
        output = self.logSoftmax(x)
        # return the output predictions
        return output