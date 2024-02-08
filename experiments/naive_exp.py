import torch
from utils.loader import get_datasets
import torch
from experiments.naive_experiment import naive_experiment
from models.ResNet50 import ResNet50
from utils.utils import exp
import torch

def run_naive_experiment(lr):
        """
        Run a naive experiment with the given learning rate.
        runs using resnet50 model on every possible combination of datasets
        logs results to folder <naive_logs_path>/RESNET/<learning_rate>/<dataset_order>

        Args:
                lr (float): The learning rate for the experiment.

        Returns:
                None
        """
        print("Collecting datasets")
        train1, test1, val1 = get_datasets("C:/Projekt/Vindr/Vindr")
        train2, test2, val2 = get_datasets("C:/Projekt/new_split")
        train3, test3, val3 = get_datasets("C:/Projekt/DDSM/DDSM")
        num_classes = 2

        orders = [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]
        names = ["Vindr_RSNA_DDSM", "Vindr_DDSM_RSNA", "RSNA_Vindr_DDSM", "RSNA_DDSM_Vindr", "DDSM_Vindr_RSNA", "DDSM_RSNA_Vindr"]

        for i in range(0, len(orders)):
                torch.cuda.empty_cache()
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model = ResNet50(num_classes)
                name_of_experiment = "RESNET/"+lr+"/"+names[i]
                experiment = exp(train1, train2, train3, test1, test2, test3, val1, val2, val3, orders[i], name_of_experiment)
                naive_experiment(model, experiment.train_set, experiment.test_set, experiment.val_set , device, name_of_experiment, 0, epochs=10, lr=lr, batch_size=32)

run_naive_experiment(0.0001)
run_naive_experiment(0.001)



