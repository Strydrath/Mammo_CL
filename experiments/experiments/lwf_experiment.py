import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from avalanche.logging import InteractiveLogger, TensorboardLogger, TextLogger
from avalanche.training import LwF
from utils.Trainer import Trainer
def icarl_experiment(model, train_set, test_set, val_set, device, name_of_experiment, exp_num, epochs=20, lr=0.001, batch_size=32, alpha=0.1, temperature=2):
    """
    Function to perform experiment using LWF strategy.

    Args:
        model (nn.Module): The neural network model.
        train_set (TensorDataset): The training dataset.
        test_set (TensorDataset): The testing dataset.
        val_set (TensorDataset): The validation dataset.
        device (str): The device to run the experiment on (e.g., 'cuda' or 'cpu').
        name_of_experiment (str): The name of the experiment.
        exp_num (int): The experiment number.
        epochs (int, optional): The number of training epochs. Defaults to 20.
        lr (float, optional): The learning rate. Defaults to 0.001.
        batch_size (int, optional): The batch size. Defaults to 32.
        alpha (float, optional): The alpha parameter. Defaults to 0.1.
        temperature (float, optional): The temperature parameter. Defaults to 2.

    Returns:
        dict: The results of the experiment.
    """
    torch.cuda.empty_cache()
    model.to(device)

    my_logger = TensorboardLogger(
        tb_log_dir="logs/LwF/" + name_of_experiment
    )

    interactive_logger = InteractiveLogger()
    log_file = open("logs/LwF/" + name_of_experiment + "/log" + str(exp_num) + ".txt", "w")
    log_file.write("epochs: " + str(epochs) + "\n")
    log_file.write("lr: " + str(lr) + "\n")
    log_file.write("batch_size: " + str(batch_size) + "\n")
    log_file.write("exp_num: " + str(exp_num) + "\n")
    log_file.write("alpha: " + str(alpha) + "\n")
    log_file.write("temperature: " + str(temperature) + "\n")
    log_file.close()
    log_file = open("logs/LwF/" + name_of_experiment + "/log" + str(exp_num) + ".txt", "a")
    text_logger = TextLogger(log_file)

    cl_strategy = LwF(
        model,
        optimizer=Adam(model.parameters(), lr=lr),
        criterion=CrossEntropyLoss(),
        alpha=alpha,
        temperature=temperature,
        train_mb_size=batch_size,
        train_epochs=epochs,
        eval_mb_size=batch_size,
        device=device,
        plugins=[
            my_logger, interactive_logger, text_logger
        ]
    
    )
    trainer = Trainer(model, train_set, test_set, val_set, device, "LwF/" + name_of_experiment)
    results = trainer.train(cl_strategy)

    return results
