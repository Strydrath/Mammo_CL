import torch
from utils.loader import get_datasets
import torch
from experiments.si_experiment import si_experiment
from models.ResNet50 import ResNet50
from utils.utils import exp
import torch

def run_si_experiment():
    """
    Runs the SI (Synaptic Intelligence) experiment using ResNet50 model on multiple datasets.

    This function collects datasets from different paths, initializes the model, and performs the SI experiment
    for different combinations of orders and lambda values.

    Logs results to folder <si_logs_path>/RESNET_FRIDAY/<dataset_order>/<lambda>.

    Returns:
        None
    """
    print("Collecting datasets")
    train1, test1, val1 = get_datasets("C:/Projekt/Vindr/Vindr")
    train2, test2, val2 = get_datasets("C:/Projekt/new_split")
    train3, test3, val3 = get_datasets("C:/Projekt/DDSM/DDSM")
    num_classes = 2
    lambdas = [0.001, 0.005, 0.001, 0.005, 0.01, 0.05,0.1,0.5]
    orders = [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]
    names = ["Vindr_RSNA_DDSM", "Vindr_DDSM_RSNA", "RSNA_Vindr_DDSM", "RSNA_DDSM_Vindr", "DDSM_Vindr_RSNA", "DDSM_RSNA_Vindr"]
    for i in range(0, len(orders)):
        for j in range(0, len(lambdas)):
            torch.cuda.empty_cache()
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = ResNet50(num_classes)
            name_of_experiment = "RESNET_FRIDAY/"+names[i]+"/"+str(lambdas[j])
            experiment = exp(train1, train2, train3, test1, test2, test3, val1, val2, val3, orders[i], name_of_experiment)
            si_experiment(model, experiment.train_set, experiment.test_set, experiment.val_set , device, name_of_experiment, j, epochs=10, lr=0.0001, batch_size=32, si_lambda=lambdas[j])

run_si_experiment()

    





