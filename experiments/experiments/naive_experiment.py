import torch
from torch.utils.data import TensorDataset
from avalanche.benchmarks.utils import AvalancheTensorDataset
from utils.loader import get_datasets
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize
from utils.di_benchmark import di_benchmark
from avalanche.training.plugins import EvaluationPlugin, EarlyStoppingPlugin
from avalanche.training.utils import adapt_classification_layer
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from models.TorchCNN import TorchCNN
from avalanche.benchmarks.classic import PermutedMNIST
from avalanche.logging import InteractiveLogger, TensorboardLogger, TextLogger
from avalanche.training import Naive
from utils.Trainer import Trainer
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
    confusion_matrix_metrics,
    timing_metrics,
    ram_usage_metrics
    
)

def naive_experiment(model, train_set, test_set, val_set, device, name_of_experiment, exp_num, epochs=20, lr=0.001, batch_size=32):
    """
    Run an experiment using naive strategy.

    Args:
        model (nn.Module): The model to be trained and evaluated.
        train_set (Dataset): The training dataset.
        test_set (Dataset): The testing dataset.
        val_set (Dataset): The validation dataset.
        device (torch.device): The device to run the experiment on.
        name_of_experiment (str): The name of the experiment.
        exp_num (int): The experiment number.
        epochs (int, optional): The number of training epochs. Defaults to 20.
        lr (float, optional): The learning rate. Defaults to 0.001.
        batch_size (int, optional): The batch size. Defaults to 32.

    Returns:
        dict: The results of the experiment.
    """
    torch.cuda.empty_cache()
    model.to(device)

    my_logger = TensorboardLogger(
        tb_log_dir="logs/naive/" + name_of_experiment
    )
    interactive_logger = InteractiveLogger()
    log_file = open("logs/naive/" + name_of_experiment + "/log" + str(exp_num) + ".txt", "w")
    log_file.write("epochs: " + str(epochs) + "\n")
    log_file.write("lr: " + str(lr) + "\n")
    log_file.write("batch_size: " + str(batch_size) + "\n")
    log_file.write("exp_num: " + str(exp_num) + "\n")
    log_file.close()
    log_file = open("logs/naive/" + name_of_experiment + "/log" + str(exp_num) + ".txt", "a")
    text_logger = TextLogger(log_file)

    evaluation_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        confusion_matrix_metrics(save_image=True, normalize="all", stream=False),
        forgetting_metrics(experience=True),
        loggers=[my_logger, interactive_logger, text_logger],
    )

    print("EXP NUM: ", exp_num)
    print("EPOCHS: ", epochs)
    print("LR: ", lr)
    print("BATCH SIZE: ", batch_size)
    print("NAME OF EXPERIMENT: ", name_of_experiment)
    print("creating strategy object")

    cl_strategy = Naive(
        model,
        Adam(model.parameters(), lr=lr),
        CrossEntropyLoss(),
        train_mb_size=batch_size,
        train_epochs=epochs,
        eval_mb_size=batch_size,
        device=device,
        evaluator=evaluation_plugin,
        eval_every=1,
        # plugins=[EarlyStoppingPlugin(patience=4,val_stream_name= "val_stream", metric_name="Loss_Exp")]
    )

    trainer = Trainer(model, train_set, test_set, val_set, device, "naive/" + name_of_experiment)
    results = trainer.train(cl_strategy)

    return results
