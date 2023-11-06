from strategies.strategy import ContinualLearningStrategy

class NaiveStrategy(ContinualLearningStrategy):
    def __init__(self, model):
        super().__init__(model)

    def train(self, datasets, epochs = 10, batch_size = 32):
        for dataset in datasets:
            self.model.fit(dataset.x_train, dataset.y_train, epochs=epochs, batch_size=batch_size)
            weights = self.model.get_weights()
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            self.model.set_weights(weights)
            history = self.model.fit(dataset.x_train,dataset.y_train, validation_data=(dataset.x_test, dataset.y_test), verbose=1, batch_size=batch_size, epochs=10)
            self.plot_acc_loss(history.history)
    def evaluate(self, model, test_loader):
        # Implement naive evaluation strategy here
        pass
