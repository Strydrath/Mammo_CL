import torch
from utils.loader import get_datasets
import torch
from models.VGG import VGGClassifier
from models.BetterCNN import BetterCNN
from models.TorchCNN import TorchCNN
from experiments.ewc_experiment import ewc_experiment
from experiments.si_experiment import si_experiment
from experiments.naive_experiment import naive_experiment

#from avalanche.models.resnet32 import ResNet, BasicBlock
from models.ResNet import ResNet
import torch
print("Collecting datasets")

train1, test1, val1 = get_datasets("C:/Projekt/Vindr/Vindr")
train2, test2, val2 = get_datasets("C:/Projekt/rsna-bcd/split")
train3, test3, val3 = get_datasets("C:/Projekt/DDSM/DDSM")

train_set = [train1, train2, train3]
test_set = [test1, test2, train3]
val_set = [val1, val2, val3]

device = torch.device("cpu")

input_shape = (512, 512)
num_classes = 2
'''
nn.Linear(self.num_features, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(16, num_classes)
'''
import torch.nn as nn
import torchvision.models as models

n = 5
model = ResNet(num_classes)
#model = BetterCNN(num_classes)
#model = VGGClassifier(num_classes)
name_of_experiment = "ResNet32"
ewc_experiment(model, train_set, test_set, val_set, device, name_of_experiment)

model = ResNet(num_classes)
#model = VGGClassifier(num_classes)
name_of_experiment = "ResNet32"
si_experiment(model, train_set, test_set, val_set, device, name_of_experiment)

model = ResNet(num_classes)
#model = VGGClassifier(num_classes)
name_of_experiment = "ResNet32"
naive_experiment(model, train_set, test_set, val_set, device, name_of_experiment)
