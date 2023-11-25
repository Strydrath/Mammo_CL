from avalanche.training.supervised.strategy_wrappers import SynapticIntelligence
from Trainer import Trainer
from loader import get_datasets
import argparse
from avalanche.training.utils import adapt_classification_layer
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from models.TorchCNN import TorchCNN
import torch


def parseOptimizer(model, optimizer):
    match optimizer:
        case "Adam":
            return Adam(model.parameters(), lr=0.001)
        case _:
            return None

def parseLossFunction(loss_function):
    match loss_function:
        case "CrossEntropyLoss":
            return CrossEntropyLoss()
        case _:
            return None

def parseStrategy(strategy,device, model, optimizer, loss_function, train_mb_size, train_epochs, eval_mb_size):
    match strategy:
        case "SI":
            return SynapticIntelligence(
                model,
                optimizer,
                loss_function,
                si_lambda = 0.0001,
                train_mb_size = train_mb_size,
                train_epochs = train_epochs,
                eval_mb_size = eval_mb_size,
                device=device,
            )
        case _:
            return None

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Command line arguments for the program')

    # Add the strategy argument
    parser.add_argument('-s', '--strategy', type=str, help='The strategy to use')

    # Add the optimizer argument
    parser.add_argument('-o', '--optimizer', type=str, help='The optimizer to use')

    # Add the loss function argument
    parser.add_argument('-l', '--loss', type=str, help='The loss function to use')

    # Parse the command line arguments
    args = parser.parse_args()

    # Access the values of the arguments
    strategy = args.strategy
    optimizer = args.optimizer
    loss_function = args.loss
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TorchCNN()
    adapt_classification_layer(model, 2, bias=False)
    optimizer = parseOptimizer(model, optimizer)
    loss_function = parseLossFunction(loss_function)
    strategy = parseStrategy(strategy, device, model, optimizer, loss_function, 2, 4, 2)

    train1, test1 = get_datasets("data/baza1")
    train2, test2 = get_datasets("data/baza2")
    train_set = [train1, train2]
    test_set = [test1, test2]

    trainer = Trainer(model, train_set, test_set, device)
    result = trainer.train(strategy)
    print(result)


if __name__ == '__main__':
    main()
