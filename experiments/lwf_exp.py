import torch
from utils.loader import get_datasets
import torch
from experiments.si_experiment import lwf_experiment
from models.ResNet50 import ResNet50
from utils.utils import exp
import torch

def run_lwf_experiment():
    """
    
    """
    print("Collecting datasets")
    train1, test1, val1 = get_datasets("C:/Projekt/Vindr/Vindr")
    train2, test2, val2 = get_datasets("C:/Projekt/new_split")
    train3, test3, val3 = get_datasets("C:/Projekt/DDSM/DDSM")
    num_classes = 2
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5]
    temperatures = [2, 3, 4, 5, 6]
    orders = [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]
    names = ["Vindr_RSNA_DDSM", "Vindr_DDSM_RSNA", "RSNA_Vindr_DDSM", "RSNA_DDSM_Vindr", "DDSM_Vindr_RSNA", "DDSM_RSNA_Vindr"]
    for order in orders:
        for alpha in alphas:
            for temperature in temperatures:
                torch.cuda.empty_cache()
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model = ResNet50(num_classes)
                name_of_experiment = "RESNET_FRIDAY/"+names[i]+"/"+str(alpha)+"/"+str(temperature)
                experiment = exp(train1, train2, train3, test1, test2, test3, val1, val2, val3, order, name_of_experiment)
                lwf_experiment(model, experiment.train_set, experiment.test_set, experiment.val_set , device, name_of_experiment, epochs=10, lr=0.0001, batch_size=32, alpha=alpha, temperature=temperature)
run_lwf_experiment()

    





