
import torch
from torch.utils.data import DataLoader
from avalanche.training.plugins import EvaluationPlugin
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize
from experiments.utils.di_benchmark import di_benchmark
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
    def __init__(self, model,  train_set, test_set, val_set, device, name_of_experiment):
        self.model = model
        self.train_set = train_set
        self.test_set = test_set
        self.val_set = val_set
        self.device = device
        self.name_of_experiment = name_of_experiment

    def train(self, strategy, callbacks=None):
        
        train_transform = transforms.Compose(
            [ transforms.Normalize((0.1307,), (0.3081,))]
        )
        test_transform = transforms.Compose(
            [ transforms.Normalize((0.1307,), (0.3081,))]
        )   
        scenario = di_benchmark(
            self.train_set, self.test_set, self.val_set, n_experiences=len(self.train_set), task_labels=True,train_transform=train_transform, eval_transform=test_transform
        )
        print("Starting experiment...")
        #print(scenario.classes_in_experience)
        adapt_classification_layer(self.model, scenario.n_classes, bias=False)
        
        cl_strategy = strategy
        print("Starting experiment...")
        results = []
        x= scenario.test_stream[0].dataset
        print("Computing accuracy on the whole test set")
        print(next(cl_strategy.model.parameters()).is_cuda)
        cl_strategy.eval(scenario.test_stream)
        i = 0
        
        for experience in scenario.train_stream:
            print("Start of experience: ", experience.current_experience)
            print("Number of examples in the test set: ", len(experience.dataset))

            cl_strategy.train(experience, eval_streams=[scenario.val_stream[i]])
            print("Training completed")
            cl_strategy.eval(scenario.test_stream)
            print("Computing accuracy on the whole test set")
            results.append(scenario.test_stream[0])
            i += 1
        
        #torch.save(cl_strategy.model, "saved_models/"+self.name_of_experiment+".pth")
        return results


