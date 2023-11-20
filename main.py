from preprocessing.preprocess import prepare_data
from strategies.naive import NaiveStrategy
from strategies.si import SynapticIntelligence
from strategies.ewc import EWCStrategy
from models.BasicCNN import create_model
from models.TorchCNN import TorchCNN
from Trainer import Trainer
import torch
from loader import get_datasets
#kocham pieski

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#dataset = prepare_data("data/baza1", categorical = True)
#dataset2 = prepare_data("data/baza2", categorical = True)

#model = create_model(512, 512, 2)
#strategy = SynapticIntelligence(model)
#strategy.train([dataset, dataset2])


#dataset = prepare_data("data/baza1", categorical = False)
#dataset2 = prepare_data("data/baza2", categorical = False)

train1, test1 = get_datasets("data/baza1")
train2, test2 = get_datasets("data/baza2")


device = torch.device("cpu")
model = TorchCNN()
trainer = Trainer(model, None, None, [train1, train2], [test1, test2], None, device)

trainer.train(10, 32, 32)