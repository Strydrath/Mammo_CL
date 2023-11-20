
import torch
from torch.utils.data import DataLoader
from avalanche.training.plugins import EvaluationPlugin
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize
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
    def __init__(self, model, optimizer, criterion, train_set, test_set, device, metrics):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_set = train_set
        self.test_set = test_set
        self.device = device
        self.metrics = metrics

    def train(self, epochs, train_mb_size, test_mb_size, callbacks=None):
        
        train_transform = transforms.Compose(
            [transforms.Normalize((0.1307,), (0.3081,))]
        )
        test_transform = transforms.Compose(
            [transforms.Normalize((0.1307,), (0.3081,))]
        )   
        scenario = ni_benchmark(
            self.train_set, self.test_set, n_experiences=2, shuffle=True, seed=1234,
            balance_experiences=True, train_transform=train_transform, eval_transform=test_transform
        )
        print("Starting experiment...")
        print(scenario.classes_in_experience)
        adapt_classification_layer(self.model, scenario.n_classes, bias=False)

        # CREATE THE STRATEGY INSTANCE (NAIVE with the Synaptic Intelligence plugin)
        cl_strategy = SynapticIntelligence(
            self.model,
            Adam(self.model.parameters(), lr=0.001),
            CrossEntropyLoss(),
            si_lambda=0.0001,
            train_mb_size=2,
            train_epochs=4,
            eval_mb_size=2,
            device=self.device,
            evaluator=evaluation_plugin,
        )
        # TRAINING LOOP
        print("Starting experiment...")
        results = []
        for experience in scenario.train_stream:
            print("Start of experience: ", experience.current_experience)
            print("Current Classes: ", experience.classes_in_this_experience)

            cl_strategy.train(experience)
            print("Training completed")

            print("Computing accuracy on the whole test set")
            results.append(cl_strategy.eval(scenario.test_stream))


    def train_epoch(self, epoch, train_loader, optimizer, criterion, device, metrics):
        self.model.train()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        train_metrics = self.test_epoch(train_loader, device, metrics)
        train_metrics["loss"] = loss.item()

        print(f"Epoch {epoch}: Train metrics: {train_metrics}")

        return train_metrics

    def test(self, test_loader, device, metrics):
        self.model.eval()

        with torch.no_grad():
            test_metrics = self.test_epoch(test_loader, device, metrics)

        print(f"Test metrics: {test_metrics}")

        return test_metrics

    def test_epoch(self, test_loader, device, metrics):
        plugin = EvaluationPlugin(metrics, reset_at='epoch', emit_at='epoch')
        plugin.before_training()

        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = self.model(inputs)
            plugin.after_forward(self.model, inputs, outputs, targets)
            plugin.after_eval_iteration(self.model, inputs, outputs, targets)

        plugin.after_training()

        return plugin.metrics_results
