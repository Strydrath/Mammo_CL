
from strategies.strategy import ContinualLearningStrategy
from keras.optimizers import Optimizer
from keras import backend as K

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import tensorflow as tf
import os
import math
import time

# elastic SGD optimizer for synaptic intelligence strategy
# it depends on it's own updates

class Elastic_SGD(Optimizer):
    def __init__(self, learning_rate=0.001, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, name="Elastic_SGD", **kwargs):
        super(Elastic_SGD, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", self._initial_decay)
        self._set_hyper("momentum", momentum)
        self._set_hyper("dampening", dampening)
        self._set_hyper("weight_decay", weight_decay)
        self.nesterov = nesterov
        
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        lr = self._decayed_lr(K.floatx())
        
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        for p, g, m in zip(params, grads, moments):
            v = self.momentum * m - lr * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

def set_lr(optimizer, lr, count):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    continue_training = True
    if count >= 10:
        continue_training = False
        print("training terminated")
    if count == 5:
        lr = lr * 0.1
        print('lr is set to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return optimizer, lr, continue_training

def terminate_protocol(since, best_acc):
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

class SynapticIntelligence(ContinualLearningStrategy):
    def __init__(self, model):
        super().__init__(model)
    
    def train(self, optimizer, datasets, num_epochs, lr = 0.001, batch_size=32):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        start_time = time.time()
        val_beat_counts = 0 
        best_acc = 0.0
        model = self.model
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

        for i,dataset in iter(datasets):
            print("Training on dataset: ", i)
            optimizer, lr, continue_training = set_lr(optimizer, lr, val_beat_counts)
            if not continue_training:
                terminate_protocol(start_time, best_acc)
                return model, best_acc
            history = model.fit(dataset.x_train, dataset.y_train, epochs=num_epochs, batch_size=batch_size)
            self.plot_acc_loss(history.history)
            with tf.GradientTape() as tape:
                outputs = model.predict(dataset.x_train)
                _, preds = tf.math.top_k(outputs, 1)
                loss = loss_fn(dataset.y_train, outputs)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
