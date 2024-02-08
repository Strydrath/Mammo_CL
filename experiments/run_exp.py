import torch
from utils.loader import get_datasets
import torch
from models.VGG import VGGClassifier
from models.BetterCNN import BetterCNN
from models.TorchCNN import TorchCNN
from experiments.ewc_experiment import ewc_experiment
from experiments.si_experiment import si_experiment
from experiments.naive_experiment import naive_experiment
from models.ResNet50 import ResNet50
import torch
print("Collecting datasets")

train1, test1, val1 = get_datasets("C:/Projekt/Vindr/Vindr")
train2, test2, val2 = get_datasets("C:/Projekt/rsna-bcd/split")
train3, test3, val3 = get_datasets("C:/Projekt/DDSM/DDSM")

train_set = [train1, train2, train3]
test_set = [test1, test2, train3]
val_set = [val1, val2, val3]

input_shape = (512, 512)
num_classes = 2


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet50(num_classes)
#model = BetterCNN(num_classes)
#model = VGGClassifier(num_classes)
name_of_experiment = "ResNet50"
ewc_experiment(model, train_set, test_set, val_set, device, name_of_experiment)

model = ResNet50(num_classes)
#model = VGGClassifier(num_classes)
name_of_experiment = "ResNet50"
si_experiment(model, train_set, test_set, val_set, device, name_of_experiment)

model = ResNet50(num_classes)
#model = VGGClassifier(num_classes)
name_of_experiment = "ResNet50"
naive_experiment(model, train_set, test_set, val_set, device, name_of_experiment)
