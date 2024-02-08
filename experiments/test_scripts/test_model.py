import tensorflow as tf
import numpy as np
from keras import layers, applications
from utils.loader import get_datasets
import matplotlib.pyplot as plt

def plot_acc_loss(history, name):
    plt.figure()
    plt.plot(history['accuracy'], label='accuracy')
    plt.plot(history['val_accuracy'], label='val_accuracy')
    plt.legend()
    plt.savefig(name+"_acc.png")
    plt.figure()
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.legend()
    plt.savefig(name+"_loss.png")

def loadDatasets(database_folder, name):
  batch_size = 3
  img_size = (512, 512)
  seed = 123
  validation_split = 0.2
  num_classes = 2

  data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
      rescale=1./255,  # Scale pixel values to the range [0, 1]
      validation_split=validation_split
  )

  train_dataset = data_generator.flow_from_directory(
      database_folder,
      target_size=img_size,
      batch_size=batch_size,
      seed=seed,
      subset='training',
      class_mode="binary"
  )

  val_dataset = data_generator.flow_from_directory(
      database_folder,
      target_size=img_size,
      batch_size=batch_size,
      seed=seed,
      subset='validation',
      class_mode="binary"
  )
  return train_dataset, val_dataset

def train(dataset_folder, name):
  # Odpowiednio zmienić ścieżki do folderu z bazą
  database_folder = "./data/baza1"
  train_dataset, val_dataset = loadDatasets(database_folder, name)

  base_model = applications.VGG16(
      include_top=False,
      weights="imagenet",
      input_tensor=None,
      input_shape=(512,512,3),
      pooling=None,
      classes=1000,
      classifier_activation="softmax",
  )

  base_model.trainable = False


  model_2 = tf.keras.models.Sequential([
      base_model,
      layers.GlobalAveragePooling2D(),
      layers.Dropout(0.2),
      layers.Dense(128, activation = 'softmax'),
      layers.Dense(2),
    ]
  )

  model_2.compile(optimizer="Adam", loss="BinaryCrossentropy", metrics=["accuracy"])

  history_2= model_2.fit(train_dataset,
                    validation_data= val_dataset,
                    epochs=10)
  
  plot_acc_loss(history_2.history, name)

def train_resnet101(database_folder, name):
  train_dataset, val_dataset = loadDatasets(database_folder, name)

  base_model = applications.ResNet101(
      include_top=False,
      weights="imagenet",
      input_tensor=None,
      input_shape=(512,512,3),
      pooling=None,
      classes=1000,
      classifier_activation="softmax",
  )

  base_model.trainable = False

  model_2 = tf.keras.models.Sequential([
    base_model,
    layers.AveragePooling2D(pool_size=(7, 7)),
    layers.Flatten(name="flatten"),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(128, activation="softmax"),
    layers.Dense(2)
  ])

  model_2.compile(optimizer="Adam", loss="BinaryCrossentropy", metrics=["accuracy"])

  history_2= model_2.fit(train_dataset,
                    validation_data= val_dataset,
                    epochs=10)
  
  plot_acc_loss(history_2.history, name)


#TU ZMIEŃ ŚCIEŻKI I NAZWY NA ODPOWIEDNIE (IDK JAK SIĘ TE BAZY NAZYWAJĄ)
#train("./data/baza1", "baza1_nazwa") 
#train("./data/baza2", "baza2_nazwa")

train_resnet101("./data/baza1", "baza1_nazwa") 
train_resnet101("./data/baza2", "baza2_nazwa")