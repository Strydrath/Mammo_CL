from types import SimpleNamespace
import datetime
def create_default_args(args_dict, additional_args=None):
    """
    Create a namespace object with default arguments.

    Args:
        args_dict (dict): A dictionary containing the default arguments.
        additional_args (Namespace, optional): Additional arguments to be added to the namespace object. Defaults to None.

    Returns:
        Namespace: A namespace object containing the default arguments and additional arguments (if provided).
    """
    print(additional_args)
    print(args_dict)
    args = SimpleNamespace()
    for k, v in args_dict.items():
        args.__dict__[k] = v
    if additional_args is not None:
        # Replace all overlapping arguments with the ones from additional_args
        for k, v in additional_args.__dict__.items():
            if k not in args_dict.keys() or v is not None:
                args.__dict__[k] = v
            
        # Add the remaining arguments from additional_args
    
    print(vars(args))
    return args

def print_args_to_file(args, file):
    """
    Print arguments to a file.
    """
    with open(file, "w") as f:
        f.write("Date and time:" + str(datetime.datetime.now()) + "\n\n")
        for k, v in vars(args).items():
            if v is not None:
                f.write(f"{k}: {v}\n")

class exp():
    """
    Represents an experiment.

    Args:
        train1 (list): List of training data for the first set.
        train2 (list): List of training data for the second set.
        train3 (list): List of training data for the third set.
        test1 (list): List of test data for the first set.
        test2 (list): List of test data for the second set.
        test3 (list): List of test data for the third set.
        val1 (list): List of validation data for the first set.
        val2 (list): List of validation data for the second set.
        val3 (list): List of validation data for the third set.
        order (list): List specifying the order of the sets.
        name_of_experiment (str): Name of the experiment.

    Attributes:
        train_set (list): List of training data in the specified order.
        test_set (list): List of test data in the specified order.
        val_set (list): List of validation data in the specified order.
    """

    def __init__(self, train1, train2, train3, test1, test2, test3, val1, val2, val3, order, name_of_experiment):
        self.train = [train1, train2, train3]
        self.test = [test1, test2, test3]
        self.val = [val1, val2, val3]
        self.order = order
        self.name_of_experiment = name_of_experiment

        self.train_set = [self.train[order[0]], self.train[order[1]], self.train[order[2]]]
        self.test_set = [self.test[order[0]], self.test[order[1]], self.test[order[2]]]
        self.val_set = [self.val[order[0]], self.val[order[1]], self.val[order[2]]]