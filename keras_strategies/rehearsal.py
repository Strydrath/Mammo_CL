from strategies.strategy import ContinualLearningStrategy
import numpy as np

class Rehearsal(ContinualLearningStrategy):
    def __init__(self, model):
        super().__init__(model)

    def shuffle_together(self, x, y):
        perm = np.random.permutation(len(x))
        x_new  = x[perm]
        y_new = y[perm]
        return (x_new, y_new)

    def get_last_data(self, datasets, perc = 0.4):
        x = None
        y = None
        for dataset in datasets:
            x_tmp, y_tmp = self.shuffle_together(dataset.x_train, dataset.y_train)
            x_tmp = x_tmp[:int(len(x_tmp)*perc)]
            y_tmp = y_tmp[:int(len(y_tmp)*perc)]

            a = np.array(x_tmp)
            print(a.shape)
            if x is None:
                x = x_tmp
                y = y_tmp
            else:
                np.concatenate((x,x_tmp))
                np.concatenate((y,y_tmp))
        a = np.array(x)
        print(a.shape)
        return (x,y)
    
    def train(self, datasets, epochs = 10, batch_size = 32):
        used_datasets = []
        for dataset in datasets:
            self.model.fit(dataset.x_train, dataset.y_train, epochs=epochs, batch_size=batch_size)
            weights = self.model.get_weights()
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            self.model.set_weights(weights)
            new_x, new_y = self.get_last_data(used_datasets)
            history = self.model.fit(new_x, new_y, verbose=1, batch_size=batch_size, epochs=10)
            self.plot_acc_loss(history.history)
            used_datasets.append(dataset)


        pass

    def evaluate(self, model, test_loader):
        # Implement rehearsal evaluation strategy here
        pass