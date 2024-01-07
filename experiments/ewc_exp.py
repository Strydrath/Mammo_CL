import torch
from utils.loader import get_datasets
import torch
from experiments.ewc_experiment import ewc_experiment
from models.ResNet50 import ResNet50
import torch
print("Collecting datasets")

train1, test1, val1 = get_datasets("C:/Projekt/DDSM/DDSM")
train2, test2, val2 = get_datasets("C:/Projekt/rsna-bcd/split")
train3, test3, val3 = get_datasets("C:/Projekt/Vindr/Vindr")

train_set = [train1, train2, train3]
test_set = [test1, test2, train3]
val_set = [val1, val2, val3]

input_shape = (512, 512)
num_classes = 2


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet50(num_classes)
name_of_experiment = "ResNet50"
ewc_experiment(model, train_set, test_set, val_set, device, name_of_experiment,7, epochs=15, lr=0.0001, batch_size=32, ewc_lambda=0.001)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet50(num_classes)
name_of_experiment = "ResNet50"
ewc_experiment(model, train_set, test_set, val_set, device, name_of_experiment,8, epochs=15, lr=0.0001, batch_size=32, ewc_lambda=0.005)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet50(num_classes)
name_of_experiment = "ResNet50"
ewc_experiment(model, train_set, test_set, val_set, device, name_of_experiment, 9, epochs=15, lr=0.0001, batch_size=32, ewc_lambda=0.001)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet50(num_classes)
name_of_experiment = "ResNet50"
ewc_experiment(model, train_set, test_set, val_set, device, name_of_experiment, 10, epochs=15, lr=0.001, batch_size=32, ewc_lambda=0.05)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet50(num_classes)
name_of_experiment = "ResNet50"
ewc_experiment(model, train_set, test_set, val_set, device, name_of_experiment, 11, epochs=15, lr=0.001, batch_size=32, ewc_lambda=0.01)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet50(num_classes)
name_of_experiment = "ResNet50"
ewc_experiment(model, train_set, test_set, val_set, device, name_of_experiment, 12, epochs=15, lr=0.01, batch_size=32, ewc_lambda=0.1)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet50(num_classes)
name_of_experiment = "ResNet50"
ewc_experiment(model, train_set, test_set, val_set, device, name_of_experiment, 13, epochs=15, lr=0.01, batch_size=32, ewc_lambda=0.5)