
import torch
from torch.utils.data import DataLoader
from avalanche.training.plugins import EvaluationPlugin
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize
from utils.di_benchmark import di_benchmark
from avalanche.benchmarks import SplitCIFAR10
from avalanche.benchmarks.generators import ni_benchmark
from avalanche.training.utils import adapt_classification_layer
from avalanche.evaluation.metrics import (
forgetting_metrics,
accuracy_metrics,
loss_metrics,
)
from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.training.supervised.strategy_wrappers import SynapticIntelligence

my_logger = TensorboardLogger(
    tb_log_dir="logs_example_SynapticIntelligence"
)

# print to stdout
interactive_logger = InteractiveLogger()

evaluation_plugin = EvaluationPlugin(
    accuracy_metrics(
        minibatch=True, epoch=True, experience=True, stream=True
    ),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    forgetting_metrics(experience=True),
    loggers=[my_logger, interactive_logger],
)





class Trainer:
    def __init__(self, model,  train_set, test_set, device):
        self.model = model
        self.train_set = train_set
        self.test_set = test_set
        self.device = device

    def train(self, strategy, callbacks=None):
        
        train_transform = transforms.Compose(
            [ transforms.Normalize((0.1307,), (0.3081,))]
        )
        test_transform = transforms.Compose(
            [ transforms.Normalize((0.1307,), (0.3081,))]
        )   
        scenario = di_benchmark(
            self.train_set, self.test_set, n_experiences=len(self.train_set), task_labels=True,train_transform=train_transform, eval_transform=test_transform
        )
        print("Starting experiment...")
        #print(scenario.classes_in_experience)
        adapt_classification_layer(self.model, scenario.n_classes, bias=False)

        # CREATE THE STRATEGY INSTANCE (NAIVE with the Synaptic Intelligence plugin)
        cl_strategy = strategy
        print("Starting experiment...")
        results = []
        x= scenario.test_stream[0].dataset
        from tqdm import tqdm
        for i, data in enumerate(tqdm(scenario.test_stream[0].dataset)):
            print(data)
        print("\nNumber of examples:", i + 1)

        for experience in scenario.train_stream:
            print("Start of experience: ", experience.current_experience)
            print("Current Classes: ", experience.classes_in_this_experience)

            cl_strategy.train(experience)
            print("Training completed")
            cl_strategy.eval(scenario.test_stream)
            print("Computing accuracy on the whole test set")
            results.append(scenario.test_stream[0])
        return results


