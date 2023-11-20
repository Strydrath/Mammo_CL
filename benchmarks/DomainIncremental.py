from avalanche.benchmarks.scenarios import ClassificationScenario

class DomainIncremental(ClassificationScenario):
    def __init__(self, train_set, test_set):
        
        super(DomainIncremental, self).__init__()