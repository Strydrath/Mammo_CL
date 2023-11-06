import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from strategies.strategy import ContinualLearningStrategy

class EWCStrategy(ContinualLearningStrategy):
    def __init__(self, model, lambda_, opt):
        super().__init__(model, opt)
        self.fisher = {}
        for layer in self.model.layers:
            for weight in layer.weights:
                self.fisher[weight.name] = tf.zeros_like(weight)
        self.prev_params = {}
        for layer in self.model.layers:
            for weight in layer.weights:
                self.prev_params[weight.name] = weight.numpy()
        self.lambda_ = lambda_

    def compute_fisher(self, data_loader):
        for weight in self.fisher:
            self.fisher[weight].assign(tf.zeros_like(self.fisher[weight]))
        for inputs, targets in data_loader:
            outputs = self.model(inputs, training=False)
            loss = tf.keras.losses.sparse_categorical_crossentropy(targets, outputs)
            grads = tf.gradients(loss, self.model.trainable_weights)
            for weight, grad in zip(self.model.trainable_weights, grads):
                self.fisher[weight.name].assign_add(tf.square(grad) / len(data_loader))

    def update_prev_params(self):
        for layer in self.model.layers:
            for weight in layer.weights:
                self.prev_params[weight.name] = weight.numpy()

    def compute_reg(self):
        reg = 0
        for weight in self.model.trainable_weights:
            reg += tf.reduce_sum(self.fisher[weight.name] * tf.square(weight - self.prev_params[weight.name]))
        return self.lambda_ * reg

    def train(self, data_loader):
        total_loss = 0
        for inputs, targets in data_loader:
            with tf.GradientTape() as tape:
                outputs = self.model(inputs, training=True)
                loss = tf.keras.losses.sparse_categorical_crossentropy(targets, outputs)
                loss += self.compute_reg()
            grads = tape.gradient(loss, self.model.trainable_weights)
            self.opt.apply_gradients(zip(grads, self.model.trainable_weights))
            total_loss += loss.numpy().mean() * inputs.shape[0]
        self.update_prev_params()
        return total_loss / len(data_loader)

    def evaluate(self, data_loader):
        total_correct = 0
        for inputs, targets in data_loader:
            outputs = self.model(inputs, training=False)
            predicted = tf.argmax(outputs, axis=1)
            total_correct += tf.reduce_sum(tf.cast(predicted == targets, tf.float32)).numpy()
        return total_correct / len(data_loader)

    def save_model(self, filepath):
        self.model.save(filepath)

    def load_model(self, filepath):
        self.model = keras.models.load_model(filepath)
