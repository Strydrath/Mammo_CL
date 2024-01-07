import torch
from utils.loader import get_datasets
import torch
from experiments.si_experiment import si_experiment
from models.ResNet50 import ResNet50
import torch
print("Collecting datasets")

train1, test1, val1 = get_datasets("C:/Projekt/new_split")
train2, test2, val2 = get_datasets("C:/Projekt/DDSM/DDSM")
train3, test3, val3 = get_datasets("C:/Projekt/Vindr/Vindr")

train_set = [train1, train2, train3]
test_set = [test1, test2, train3]
val_set = [val1, val2, val3]

input_shape = (512, 512)
num_classes = 2


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet50(num_classes)
name_of_experiment = "ResNet50"
si_experiment(model, train_set, test_set, val_set, device, name_of_experiment,14, epochs=10, lr=0.001, batch_size=32, si_lambda=0.1)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet50(num_classes)
name_of_experiment = "ResNet50"
si_experiment(model, train_set, test_set, val_set, device, name_of_experiment,15, epochs=10, lr=0.0001, batch_size=32, si_lambda=0.01)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet50(num_classes)
name_of_experiment = "ResNet50"
si_experiment(model, train_set, test_set, val_set, device, name_of_experiment, 16, epochs=10, lr=0.0001, batch_size=32, si_lambda=0.05)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet50(num_classes)
name_of_experiment = "ResNet50"
si_experiment(model, train_set, test_set, val_set, device, name_of_experiment, 17, epochs=10, lr=0.0001, batch_size=32, si_lambda=0.005)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet50(num_classes)
name_of_experiment = "ResNet50"
si_experiment(model, train_set, test_set, val_set, device, name_of_experiment, 18, epochs=10, lr=0.0001, batch_size=32, si_lambda=0.001)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet50(num_classes)
name_of_experiment = "ResNet50"
si_experiment(model, train_set, test_set, val_set, device, name_of_experiment, 19, epochs=10, lr=0.0001, batch_size=32, si_lambda=0.5)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet50(num_classes)
name_of_experiment = "ResNet50"
si_experiment(model, train_set, test_set, val_set, device, name_of_experiment, 20, epochs=10, lr=0.0001, batch_size=32, si_lambda=0.003)