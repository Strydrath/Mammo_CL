# Continual learning for mammography classification

Course project from 2023/2024 at Gdańsk University of Technology, Department of Biomedical Engineering, supervised by Jacek Rumiński, PhD.

The code runs domain-incremental continual-learning experiments on mammography scans from three datasets: Vindr, DDSM, and RSNA. It uses the Avalanche library, with Avalanche classes for domain-incremental learning that also apply outside this medical setting.

Logs are written as text files and as TensorBoard event files. The `logs/` directory holds the runs from this project, and those event files are why the repository is large.

## Installation

Python 3.10.2.

```bash
pip install -r requirements.txt
```

## Usage

Run one experiment from the command line:

```bash
python main.py --strategy STRATEGY --name experiment_name
```

Or run a batch with `experiments/ewc_exp.py`, `experiments/naive_exp.py`, or `experiments/si_exp.py`.

### Arguments

| Flag | Meaning |
| --- | --- |
| `-s`, `--strategy` | Continual-learning strategy |
| `-o`, `--optimizer` | Optimizer |
| `-l`, `--loss` | Loss function |
| `-e`, `--epochs` | Number of training epochs |
| `-lr`, `--learning_rate` | Learning rate |
| `-tms`, `--train_mb_size` | Training minibatch size |
| `-ems`, `--eval_mb_size` | Evaluation minibatch size |
| `-si_lambda`, `--si_lambda` | Lambda for Synaptic Intelligence |
| `-si_eps`, `--si_eps` | Epsilon for Synaptic Intelligence |
| `-ewc_lambda`, `--ewc_lambda` | Lambda for EWC |
| `-ewc_mode`, `--ewc_mode` | EWC mode |
| `-ewc_decay`, `--ewc_decay` | EWC decay |
| `-dropout`, `--dropout` | Dropout |
| `-seed`, `--seed` | Random seed |
| `-hidden_size`, `--hidden_size` | Hidden size |
| `-hidden_layers`, `--hidden_layers` | Number of hidden layers |
| `-name`, `--name` | Experiment name used in the logs |
| `-db_o`, `--db_order` | Dataset order: `V` Vindr, `R` RSNA, `D` DDSM. Example: `VDR` |

## Layout

- `preprocessing/` converts DICOM files to JPEG.
- `experiments/` runs the continual-learning experiments. `experiments/utils/` extends Avalanche for domain-incremental scenarios.
- `keras_strategies/` is a separate Keras baseline for Synaptic Intelligence, EWC, and a naive strategy. The main experiments do not use it.
- `logs/` stores logs from the experiments that were run.

## Team

- Marianna Jucewicz
- Radosław Karkoszka
- Aleksandra Nadzieja

Supervisor: Jacek Rumiński, PhD
