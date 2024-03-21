from avalanche.training.supervised.strategy_wrappers import SynapticIntelligence
from avalanche.training import EWC
from avalanche.training.supervised.strategy_wrappers import LwF
from avalanche.training.supervised.icarl import ICaRL
from avalanche.training.plugins import EvaluationPlugin
from avalanche.logging import InteractiveLogger, TensorboardLogger, TextLogger
from experiments.utils.Trainer import Trainer
from experiments.utils.loader import get_datasets
import argparse
from avalanche.training.utils import adapt_classification_layer
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from experiments.models.ICARL_models import Feature_Extractor, Classifier
import torch
from experiments.utils.utils import create_default_args, exp, print_args_to_file
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
    confusion_matrix_metrics
)

from experiments.models.ResNet50 import ResNet50

def parseOptimizer(model, optimizer):
    """
    Parses the optimizer string and returns the corresponding optimizer object.

    Args:
        model (torch.nn.Module): The model to optimize.
        optimizer (str): The optimizer string.

    Returns:
        torch.optim.Optimizer or None: The optimizer object or None if the optimizer string is not recognized.
    """
    match optimizer:
        case "Adam":
            return Adam(model.parameters(), lr=0.001)
        case _:
            return Adam(model.parameters(), lr=0.001)

def parseLossFunction(loss_function):
    """
    Parses the loss function string and returns the corresponding loss function object.

    Args:
        loss_function (str): The loss function string.

    Returns:
        torch.nn.Module or None: The loss function object or None if the loss function string is not recognized.
    """
    match loss_function:
        case "CrossEntropyLoss":
            return CrossEntropyLoss()
        case _:
            return CrossEntropyLoss()

def createStrategyArgs(strategy, overrideArgs):
    """
    Creates the default strategy arguments based on the given strategy string and overrides them with the provided arguments.

    Args:
        strategy (str): The strategy string.
        overrideArgs (dict): The dictionary of arguments to override the default arguments.

    Returns:
        dict: The strategy arguments.
    """
    match strategy:
        case "NAIVE":
            return create_default_args({'cuda': 0, 
                                        'epochs': 10, 
                                        'learning_rate': 0.001, 
                                        'train_mb_size': 256, 
                                        'seed': None,
                                        'db_order': 'VDR',
                                        'name': 'NAIVE_EXP'
                                        },
                                        overrideArgs)
        case "SI":
            return create_default_args({'cuda': 0, 
                                        'si_lambda': 10, 
                                        'si_eps': 0.1, 
                                        'epochs': 10,
                                        'learning_rate': 0.001, 
                                        'train_mb_size': 32, 
                                        'seed': None,
                                        'db_order': 'VDR',
                                        'name': 'SI_EXP'
                                        },
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
                                        'train_mb_size': 32,
                                        'seed': None,
                                        'db_order': 'VDR',
                                        'name': 'NAIVE_EXP'},
                                        overrideArgs)
        case "LWF":
            return create_default_args({'cuda': 0,
                                        'alpha': 0.1, 
                                        'temperature': 2, 
                                        'epochs': 10, 
                                        'learning_rate': 0.001, 
                                        'train_mb_size': 32, 
                                        'seed': None,
                                        'db_order': 'VDR',
                                        'name': 'LWF_EXP'},
                                        overrideArgs)
        case "ICARL":
            return create_default_args({'cuda': 0, 
                                        'epochs': 10, 
                                        'learning_rate': 0.001, 
                                        'train_mb_size': 256, 
                                        'seed': None,
                                        'db_order': 'VDR',
                                        'fixed_memory': False,
                                        'name': 'ICARL_EXP'},
                                        overrideArgs)
        case _:
            return create_default_args({'cuda': 0, 
                                        'epochs': 10, 
                                        'learning_rate': 0.001, 
                                        'train_mb_size': 256, 
                                        'seed': None}
                                        )

def createParser():
    """
    Creates the argument parser for the program.

    Returns:
        argparse.ArgumentParser: The argument parser.
    """
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
    parser.add_argument('-name', '--name', type=str, help='The name of the experiment')
    parser.add_argument('-db_o', '--db_order', type=str, help='The order of the datasets')
    parser.add_argument('-alpha', '--alpha', type=float, help='The alpha value for LwF')
    parser.add_argument('-temperature', '--temperature', type=float, help='The temperature value for LwF')
    parser.add_argument('-f_m', '--fixed_memory', type=bool, help='If memory is fixed')
    return parser

"""
Arguments allowed by the program:
    -s, --strategy: The strategy to use
    -o, --optimizer: The optimizer to use
    -l, --loss: The loss function to use
    -e, --epochs: The number of epochs to train for
    -lr, --learning_rate: The learning rate to use
    -tms, --train_mb_size: The training minibatch size
    -ems, --eval_mb_size: The evaluation minibatch size
    -si_lambda, --si_lambda: The lambda value for SI
    -si_eps, --si_eps: The epsilon value for SI
    -ewc_lambda, --ewc_lambda: The lambda value for EWC
    -ewc_mode, --ewc_mode: The mode for EWC
    -ewc_decay, --ewc_decay: The decay factor for EWC
    -dropout, --dropout: The dropout value for EWC
    -seed, --seed: The seed to use
    -hidden_size, --hidden_size: The hidden size for EWC
    -hidden_layers, --hidden_layers: The number of hidden layers for EWC
    -name, --name: The name of the experiment used for logs
    -db_o, --db_order: The order of the datasets (V -for Vindr, R - for RSNA, D - for DDSM example: "VDR")
    -alpha, --alpha: The alpha value for LwF
    -temperature, --temperature: The temperature value for LwF
    -f_m, --fixed_memory: The fixed memory size for ICARL

"""

def parseStrategy(strategy, device, model, optimizer, loss_function, args):
    """
    Parses the strategy string and returns the corresponding strategy object.

    Args:
        strategy (str): The strategy string.
        device (torch.device): The device to use for training.
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        loss_function (torch.nn.Module): The loss function to use.
        args (argparse.Namespace): The parsed command line arguments.

    Returns:
        avalanche.training.BaseStrategy or None: The strategy object or None if the strategy string is not recognized.
    """
    log_dir = "logs_TEST/"+strategy+"/"+args.db_order
    if(args.si_lambda):
        log_dir += "/"+str(args.si_lambda)
    if(args.ewc_lambda):
        log_dir += "/"+str(args.ewc_lambda)
    if(args.alpha):
        log_dir += "/"+str(args.alpha)
    if(args.temperature):
        log_dir += "/"+str(args.temperature)
    
    my_logger = TensorboardLogger(
    tb_log_dir=log_dir
    )
    interactive_logger = InteractiveLogger()
    print_args_to_file(args, log_dir+"/args.txt")
    log_file = open(log_dir+"/log.txt", "w")
    text_logger = TextLogger(log_file)
    evaluation_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=True, epoch=True, experience=True, stream=True
        ),
        confusion_matrix_metrics(save_image=True, normalize="all", stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True),
        loggers=[my_logger, interactive_logger, text_logger],
    )
    if strategy == "ICARL":
        feature_extractor = Feature_Extractor()
        classifier = Classifier()
    match strategy:
        case "SI":
            return SynapticIntelligence(
                model,
                optimizer,
                loss_function,
                si_lambda = args.si_lambda,
                train_mb_size = args.train_mb_size,
                train_epochs = args.epochs,
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
        case "LWF":
            return LwF(
                model, 
                optimizer, 
                loss_function, 
                alpha=args.alpha, 
                temperature=args.temperature,
                train_mb_size=args.train_mb_size, 
                train_epochs=args.epochs, 
                eval_mb_size=args.eval_mb_size,
                device=device, 
                evaluator=evaluation_plugin)
        case "ICARL":
            return ICaRL(
                feature_extractor,
                classifier,
                optimizer = optimizer, 
                train_mb_size=args.train_mb_size, 
                train_epochs=args.epochs,
                eval_mb_size=args.eval_mb_size,
                memory_size=2000,
                buffer_transform=None,
                
                fixed_memory=False,
                device=device,
                evaluator=evaluation_plugin)
        case _:
            return None

def parse_database_order(db_order):
    match db_order:
        case "VRD":
            return [0,1,2]
        case "VDR":
            return [0,2,1]
        case "RVD":
            return [1,0,2]
        case "RDV":
            return [1,2,0]
        case "DVR":
            return [2,0,1]
        case "DRV":
            return [2,1,0]
        case _:
            return None



def main():
    """
    The main function of the program.
    """
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
    
    model = ResNet50(num_classes=2 )

    #adapt_classification_layer(model, 2, bias=False)
    optimizer = parseOptimizer(model, optimizer)
    loss_function = parseLossFunction(loss_function)
    strategy = parseStrategy(strategy, device, model, optimizer, loss_function,args)
    

    train1, test1, val1 = get_datasets("C:/Users/SPECTRE/Desktop/STUDIA/PracaMagisterska/Mammo_CL/db")
    train2, test2, val2 = get_datasets("C:/Users/SPECTRE/Desktop/STUDIA/PracaMagisterska/Mammo_CL/db")
    train3, test3, val3 = get_datasets("C:/Users/SPECTRE/Desktop/STUDIA/PracaMagisterska/Mammo_CL/db")
    
    """
    train1, test1, val1 = get_datasets("C:/Projekt/Vindr/Vindr")
    train2, test2, val2 = get_datasets("C:/Projekt/new_split")
    train3, test3, val3 = get_datasets("C:/Projekt/DDSM/DDSM")
    """
    order = parse_database_order(args.db_order)
    experiment = exp(train1, train2, train3, test1, test2, test3, val1, val2, val3, order, args.name)
    trainer = Trainer(model, experiment.train_set, experiment.test_set, experiment.val_set, device, args.name)
    result = trainer.train(strategy)
    print(result)


if __name__ == '__main__':
    main()
