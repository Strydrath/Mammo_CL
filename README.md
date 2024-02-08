# Continual learning in the problem of classifying breast mammographic images

Code for the Project created in 2023/2024 on Gdansk University of Technology, Department of Biomedical Engineering under supervision of Jacek Ruminski PhD.

Project allows user to run experiments in regard of continual learning in incremental domain setting.
Datasets used for this project are Mammography scans from 3 databases: Vindr, DDSM and RSNA.

Project uses Avalanche library and provides implementation of avalanche classes for DOmain Incremental Continual Learning.
Logs are created as txt files as well as tensorboard ones.

Code is documented in comments of each file

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)

## Installation

Project requires Python version 3.10.2.
All required dependencies can be installed using pip command:
```
pip install -r requirements.txt
```
No further installation is needed.

## Usage

Project 
1. Using the command line arguments:

Run the main file with command line arguments to run specific experiments.



Example: python main.py --experiment experiment_name

Arguments allowed by the script:
```
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
    -db_o, --db_order: The order of the datasets (V -for Vindr, R - for new_split, D - for DDSM example: "VDR")
```

2. Running specific experiment files:

Run one of the following files: ewc_exp.py, naive_exp.py, or si_exp.py to run a whole batch of experiments.
## Features

Project contains 3 main parts:
* `/preprocessing` - code used to process *.dicom files into *.jpg files. 
* `/experiments` - code allowing for running Continual Learning experiments
especially `/experiments/utils` - files extanding Avalanche classes allowing for Domain Incremental Continual Learning experiments - classes are universal and can be used for any Domain Incremental scenario - not only related to medicine.
* `/keras_strategies` - files with keras implementation of Synaptic Intelligence, EWC and Naive strategies, not used in the rest of the project - baseline of how the strategies work.

There are also logs generated during experiments conducted by the team - they are in the `/logs` folder.



## Contact

Team:
* Marianna Jucewicz, s180377@student.pg.edu.pl
* Radosław Karkoszka, s197124@student.pg.edu.pl
* Aleksandra Nadzieja, s175237@student.pg.edu.pl

Supervisor:
Ph.D. Jacek Rumiński, jacrumin@pg.edu.pl


