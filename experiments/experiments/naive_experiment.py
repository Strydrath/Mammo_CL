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
from avalanche.logging import InteractiveLogger, TensorboardLogger
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

def naive_experiment(model, train_set, test_set, val_set, device, name_of_experiment):
    torch.cuda.empty_cache()
    model.to(device)



    my_logger = TensorboardLogger(
        tb_log_dir="logs/naive/"+name_of_experiment
    )

    interactive_logger = InteractiveLogger()
    evaluation_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=True, epoch=True, experience=True, stream=True
        ),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True),
        loggers=[my_logger, interactive_logger],
    )
    print("creating strategy object")
    cl_strategy = Naive(
        model,
        Adam(model.parameters(), lr=0.001),
        CrossEntropyLoss(),
        train_mb_size=32,
        train_epochs=20,
        eval_mb_size=32,
        device=device,
        evaluator=evaluation_plugin,
        eval_every=1,
        #plugins=[EarlyStoppingPlugin(patience=4,val_stream_name= "val_stream", metric_name="Loss_Exp")]
    )

    trainer = Trainer(model, train_set, test_set, val_set, device, "naive/"+name_of_experiment)
    results = trainer.train(cl_strategy)
