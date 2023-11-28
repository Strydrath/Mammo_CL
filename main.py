from avalanche.training.supervised.strategy_wrappers import SynapticIntelligence
from avalanche.training import EWC
from avalanche.training.plugins import EvaluationPlugin
from avalanche.logging import InteractiveLogger, TensorboardLogger
from Trainer import Trainer
from loader import get_datasets
import argparse
from avalanche.training.utils import adapt_classification_layer
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from models.TorchCNN import TorchCNN
import torch
from loader import get_datasets
from utils import create_default_args
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)

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

def createStrategyArgs(strategy, overrideArgs):
    match strategy:
        case "NAIVE":
            return create_default_args({'cuda': 0, 
                                        'epochs': 10, 
                                        'learning_rate': 0.001, 
                                        'train_mb_size': 256, 
                                        'seed': None},
                                        overrideArgs)
        case "SI":
            return create_default_args({'cuda': 0, 
                                        'si_lambda': 10, 
                                        'si_eps': 0.1, 
                                        'epochs': 10,
                                        'learning_rate': 0.001, 
                                        'train_mb_size': 256, 
                                        'seed': None},
                                        overrideArgs)    
        case "EWC":
            return create_default_args({'cuda': 0, 
                                        'ewc_lambda': 1, 
                                        'hidden_size': 512,
                                        'hidden_layers': 1, 
                                        'epochs': 10, 
                                        'dropout': 0,
                                        'ewc_mode': 'separate', 
                                        'ewc_decay': None,
                                        'learning_rate': 0.001, 
                                        'train_mb_size': 256,
                                        'seed': None},
                                        overrideArgs)
        case _:
            return create_default_args({'cuda': 0, 
                                        'epochs': 10, 
                                        'learning_rate': 0.001, 
                                        'train_mb_size': 256, 
                                        'seed': None}
                                        )

def createParser():
    parser = argparse.ArgumentParser(description='Command line arguments for the program')
    parser.add_argument('-s', '--strategy', type=str, help='The strategy to use')
    parser.add_argument('-o', '--optimizer', type=str, help='The optimizer to use')
    parser.add_argument('-l', '--loss', type=str, help='The loss function to use')
    parser.add_argument('-e', '--epochs', type=int, help='The number of epochs to train for')
    parser.add_argument('-lr', '--learning_rate', type=float, help='The learning rate to use')
    parser.add_argument('-tms', '--train_mb_size', type=int, help='The training minibatch size')
    parser.add_argument('-ems', '--eval_mb_size', type=int, help='The evaluation minibatch size')
    parser.add_argument('-si_lambda', '--si_lambda', type=float, help='The lambda value for SI')
    parser.add_argument('-si_eps', '--si_eps', type=float, help='The epsilon value for SI')
    parser.add_argument('-ewc_lambda', '--ewc_lambda', type=float, help='The lambda value for EWC')
    parser.add_argument('-ewc_mode', '--ewc_mode', type=str, help='The mode for EWC')
    parser.add_argument('-ewc_decay', '--ewc_decay', type=float, help='The decay factor for EWC')
    parser.add_argument('-dropout', '--dropout', type=float, help='The dropout value for EWC')
    parser.add_argument('-seed', '--seed', type=int, help='The seed to use')
    parser.add_argument('-hidden_size', '--hidden_size', type=int, help='The hidden size for EWC')
    parser.add_argument('-hidden_layers', '--hidden_layers', type=int, help='The number of hidden layers for EWC')
    return parser



def parseStrategy(strategy,device, model, optimizer, loss_function, args):
    my_logger = TensorboardLogger(
    tb_log_dir="logs_example_"+strategy
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
    match strategy:
        case "SI":
            return SynapticIntelligence(
                model,
                optimizer,
                loss_function,
                si_lambda = args.si_lambda,
                train_mb_size = args.train_mb_size,
                train_epochs = args.train_epochs,
                eval_mb_size = args.eval_mb_size,
                device=device,
                evaluator=evaluation_plugin
            )
        case "EWC":
            return EWC(
                model, 
                optimizer, 
                loss_function,
                ewc_lambda=args.ewc_lambda, 
                mode=args.ewc_mode, 
                decay_factor=args.ewc_decay,
                train_mb_size=args.train_mb_size, 
                train_epochs=args.epochs, 
                eval_mb_size=args.eval_mb_size,
                device=device, 
                evaluator=evaluation_plugin)
        case _:
            return None

def main():
    # Create the argument parser
    parser = createParser()
    # Parse the command line arguments
    args = parser.parse_args()

    # Access the values of the arguments
    strategy = args.strategy
    optimizer = args.optimizer
    loss_function = args.loss
    
    args = createStrategyArgs(strategy, args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TorchCNN()
    adapt_classification_layer(model, 2, bias=False)
    optimizer = parseOptimizer(model, optimizer)
    loss_function = parseLossFunction(loss_function)
    strategy = parseStrategy(strategy, device, model, optimizer, loss_function,args)

    train1, test1 = get_datasets("data/baza1")
    train2, test2 = get_datasets("data/baza2")
    train_set = [train1, train2]
    test_set = [test1, test2]

    trainer = Trainer(model, train_set, test_set, device)
    result = trainer.train(strategy)
    print(result)


if __name__ == '__main__':
    main()
