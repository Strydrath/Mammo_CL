from keras.models import Sequential
import matplotlib.pyplot as plt

class ContinualLearningStrategy:
    model: Sequential
    def __init__(self, model):
        self.model = model
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def plot_acc_loss(self, history):
        plt.figure(figsize=(20, 6))

        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='accuracy')
        plt.plot(history['val_accuracy'], label='val_accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='loss')
        plt.plot(history['val_loss'], label='val_loss')
        plt.legend()
    def train(self, data):
        pass
    
    def predict(self, data):
        pass
