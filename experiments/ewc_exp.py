import torch
from utils.loader import get_datasets
import torch
from experiments.ewc_experiment import ewc_experiment
from models.ResNet50 import ResNet50
import torch
print("Collecting datasets")

train1, test1, val1 = get_datasets("C:/Projekt/Vindr/Vindr")
train2, test2, val2 = get_datasets("C:/Projekt/new_split")
train3, test3, val3 = get_datasets("C:/Projekt/DDSM/DDSM")

class exp():
    def __init__(self, train1, train2, train3, test1, test2, test3, val1, val2, val3, order, name_of_experiment):
        self.train = [train1, train2, train3]
        self.test = [test1, test2, test3]
        self.val = [val1, val2, val3]
        self.order = order
        self.name_of_experiment = name_of_experiment

        self.train_set = [self.train[order[0]], self.train[order[1]], self.train[order[2]]]
        self.test_set = [self.test[order[0]], self.test[order[1]], self.test[order[2]]]
        self.val_set = [self.val[order[0]], self.val[order[1]], self.val[order[2]]]

num_classes = 2

lambdas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05,0.1,0.5]
orders = [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]
names = ["Vindr_RSNA_DDSM", "Vindr_DDSM_RSNA", "RSNA_Vindr_DDSM", "RSNA_DDSM_Vindr", "DDSM_Vindr_RSNA", "DDSM_RSNA_Vindr"]
for i in range(0, len(orders)):
    for j in range(0, len(lambdas)):
        torch.cuda.empty_cache()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = ResNet50(num_classes)
        name_of_experiment = "RESNET_FRIDAY_2/"+names[i]+"/"+str(lambdas[j])
        experiment = exp(train1, train2, train3, test1, test2, test3, val1, val2, val3, orders[i], name_of_experiment)
        ewc_experiment(model, experiment.train_set, experiment.test_set, experiment.val_set , device, name_of_experiment, j, epochs=10, lr=0.0001, batch_size=32, ewc_lambda=lambdas[j])

    





