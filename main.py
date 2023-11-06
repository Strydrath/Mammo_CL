from preprocessing.preprocess import prepare_data
from strategies.naive import NaiveStrategy
from strategies.si import SynapticIntelligence
from models.BasicCNN import create_model

dataset = prepare_data("data/baza1", categorical = True)
dataset2 = prepare_data("data/baza2", categorical = True)

model = create_model(512, 512, 2)
strategy = SynapticIntelligence(model)
strategy.train([dataset, dataset2])