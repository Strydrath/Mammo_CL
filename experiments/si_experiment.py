import torch
from torch.utils.data import TensorDataset
from avalanche.benchmarks.utils import AvalancheTensorDataset
from utils.loader import get_datasets
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize
from utils.di_benchmark import di_benchmark
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.utils import adapt_classification_layer
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from models.TorchCNN import TorchCNN
from avalanche.benchmarks.classic import PermutedMNIST
from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.training import SynapticIntelligence
from utils.Trainer import Trainer
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)


print("Collecting datasets")
# ZMIENIĆ ŚCIEŻKĘ NA FOLDER Z PIERWSZĄ BAZĄ
train1, test1 = get_datasets("experiments/data/baza1")
# ZMIENIĆ ŚCIEŻKĘ NA FOLDER Z DRUGĄ BAZĄ
train2, test2 = get_datasets("experiments/data/baza2")

train_set = [train1, train2]
test_set = [test1, test2]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



model = TorchCNN()
adapt_classification_layer(model, 2, bias=False)

my_logger = TensorboardLogger(
    tb_log_dir="logs_example_NAIVE"
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
cl_strategy = SynapticIntelligence(
    model,
    Adam(model.parameters(), lr=0.001),
    CrossEntropyLoss(),
    si_lambda=0.0001,
    train_mb_size=64,
    train_epochs=10,
    eval_mb_size=64,
    device=device,
    evaluator=evaluation_plugin,
)

trainer = Trainer(model, train_set, test_set,device)
results = trainer.train(cl_strategy)
