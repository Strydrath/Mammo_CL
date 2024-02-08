from copy import deepcopy
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras import layers
from strategies.strategy import ContinualLearningStrategy

class EWCStrategy(ContinualLearningStrategy):
    def __init__(self, model):
        super().__init__(model)

    def train_epoch(self, train_data, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices(train_data)
        dataset = dataset.shuffle(len(train_data[0])).batch(batch_size)

        for inputs, labels in dataset:
            with tf.GradientTape() as tape:
                outputs = self.model(inputs)
                loss = self.model.compiled_loss(labels, outputs)

            gradients = tape.gradient(loss, self.model.trainable_weights)
            self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        return loss
    
    def report(self, epoch, dataset, batch_size):
        loss, accuracy = self.model.evaluate(dataset.x_train, dataset.y_train, verbose=0,
                                    batch_size=batch_size)
        loss2, accuracy2 = self.model.evaluate(dataset.x_test, dataset.y_test, verbose=0,
                                    batch_size=batch_size)
        print("Epoch: ",epoch + 1, "Train Set:  Accuracy: ","{:.4f}".format(accuracy),
            "Loss: ","{:.4f}".format(loss), "Validation Set: Accuracy: ",
            "{:.4f}".format(accuracy2), "Loss: ", "{:.4f}".format(loss2))
        return (accuracy,loss, accuracy2 ,loss2)

    def compile_model(self,model, learning_rate, regularisers=None):
        def custom_loss(y_true, y_pred):
            loss = sparse_categorical_crossentropy(y_true, y_pred)
            if regularisers is not None:
                for fun in regularisers:
                    loss += fun(model)
            return loss
        self.model.compile(
            loss=custom_loss,
            optimizer=Adam(learning_rate=learning_rate),
            metrics=["accuracy"]
        )
    def fisher_matrix(self, dataset, samples):
        inputs, labels = dataset
        weights = self.model.trainable_weights
        variance = [tf.zeros_like(tensor) for tensor in weights]

        for _ in range(samples):
            index = np.random.randint(len(inputs))
            data = inputs[index]
            data = tf.expand_dims(data, axis=0)

            with tf.GradientTape() as tape:
                output = self.model(data)
                log_likelihood = tf.math.log(output)

            gradients = tape.gradient(log_likelihood, weights)

            variance = [var + (grad ** 2) for var, grad in zip(variance, gradients)]

        fisher_diagonal = [tensor / samples for tensor in variance]
        return fisher_diagonal


    def ewc_loss(self, lam, dataset, samples):
        optimal_weights = deepcopy(self.model.trainable_weights)
        fisher_diagonal = self.fisher_matrix(self.model, dataset, samples)

        def loss_fn(new_model):
            # sum [(lambda / 2) * F * (current weights - optimal weights)^2]
            loss = 0
            current = new_model.trainable_weights
            for f, c, o in zip(fisher_diagonal, current, optimal_weights):
                loss += tf.reduce_sum(f * ((c - o) ** 2))

            return loss * (lam / 2)

        return loss_fn

    def train(self, datasets, learning_rate = 0.001, epochs = 10, batch_size = 32, ewc_lambda = 1, ewc_samples = 100):
        self.compile_model(self.model, learning_rate)
        regularisers = []
        histories = []

        for dataset in datasets:
            history = {
                'accuracy': [],
                'val_accuracy': [],
                'loss': [],
                'val_loss': []
            }
            inputs, labels = dataset.x_train, dataset.y_train

            for epoch in range(epochs):
                loss = self.train_epoch((inputs, labels), batch_size)
                acc,loss, val_acc, val_loss = self.report( epoch, dataset, batch_size)
                history['loss'].append(loss)
                history['accuracy'].append(acc)
                history['val_accuracy'].append(val_acc)
                history['val_loss'].append(val_loss)
            histories.append(history)
            regularisers.append(self.ewc_loss(ewc_lambda, (inputs, labels),ewc_samples))
            self.compile_model(self.model, learning_rate, regularisers=regularisers)
        for history in histories:
            self.plot_acc_loss(history)
        return (histories,self.model)

    def evaluate(self, data_loader):
        pass

    def save_model(self, filepath):
        self.model.save(filepath)

    def load_model(self, filepath):
        self.model = keras.models.load_model(filepath)
