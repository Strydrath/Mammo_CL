import torch
from torch.utils.data import TensorDataset
from avalanche.benchmarks.utils import AvalancheTensorDataset
from loader import get_datasets
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize
from avalanche.benchmarks.generators import (ni_benchmark, dataset_benchmark, ni_scenario)
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.utils import adapt_classification_layer
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from models.TorchCNN import TorchCNN
from avalanche.benchmarks.classic import PermutedMNIST
from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.training.supervised.strategy_wrappers import SynapticIntelligence
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)


train1, test1 = get_datasets("data/baza1")
train2, test2 = get_datasets("data/baza2")
train_set = [train1, train2]
test_set = [test1, test2]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose(
            [ transforms.Normalize((0.1307,), (0.3081,))]
        )
test_transform = transforms.Compose(
    [ transforms.Normalize((0.1307,), (0.3081,))]
)   
pattern = [list(range(0,len(train1))),list(range(len(train1),len(train1)+len(train2)))]


scenario = ni_benchmark(
    train_set, test_set, n_experiences=2, task_labels=True,
    balance_experiences=True, train_transform=train_transform, eval_transform=test_transform
)

model = TorchCNN()
# train_ass = scenario.train_exps_patterns_assignment
adapt_classification_layer(model, 2, bias=False)

my_logger = TensorboardLogger(
    tb_log_dir="logs_example_SynapticIntelligence"
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

# CREATE THE STRATEGY INSTANCE (NAIVE with the Synaptic Intelligence plugin)
cl_strategy = SynapticIntelligence(
    model,
    Adam(model.parameters(), lr=0.001),
    CrossEntropyLoss(),
    si_lambda=0.0001,
    train_mb_size=2,
    train_epochs=4,
    eval_mb_size=2,
    device=device,
    evaluator=evaluation_plugin,
)

# TRAINING LOOP
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
    cl_strategy.eval(scenario.test_stream[0])
    print("Computing accuracy on the whole test set")
    #results.append(scenario.test_stream[0])