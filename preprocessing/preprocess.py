from dataclasses import dataclass

import numpy as np
from PIL import Image
import os
import pathlib
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

@dataclass
class Dataset:
  x_train : np.ndarray
  y_train : np.ndarray
  x_test : np.ndarray
  y_test : np.ndarray


def prepare_X_y(files_paths, files_labels):
  X = []
  y = []

  for path in files_paths:
    img = Image.open(path)
    img.load()
    img = img.resize((512, 512)) # na potrzeby danych testowych. przetworzone obrazy powinny
                                 #już mieć ustawione odpowiednie wymiary

    img_X = np.asarray(img, dtype=np.int16)
    X.append(img_X)

  X = np.array(X)
  y = np.array(files_labels)

  return X, y

def preprocess(dataset, labels, categorical = False):
  num_classes = len(labels)

  train_hlp= list(map(lambda x: labels.index(x), dataset.y_train))
  train_hlp = np.array(train_hlp)
  dataset.y_train = train_hlp

  test_hlp = list(map(lambda x: labels.index(x), dataset.y_test))
  test_hlp = np.array(test_hlp)
  dataset.y_test = test_hlp

  if categorical:
    dataset.y_train = to_categorical(dataset.y_train, num_classes)
    dataset.y_test = to_categorical(dataset.y_test, num_classes)

  return dataset

def basic_preparation(folder):
  p = pathlib.Path(folder)
  dirs = p.iterdir()

  labels = []
  labels_folders = []

  for x in dirs:
    labels.append(x.parts[-1])
    labels_folders.append(x)

  num_classes = len(labels)
  labels.sort()
  print(f"Available labels: {labels}")

  files_paths = []
  files_labels = []

  for root, dirs, files in os.walk(folder):
    p = pathlib.Path(root)
    if p not in labels_folders:
      continue
    for file in files:
      files_paths.append(root + '/' + file)
      files_labels.append(p.parts[-1])

  print(files_labels)

  print(f"Number of all images: {len(files_labels)}")
  x, y = prepare_X_y(files_paths,files_labels)
  return x, y, labels

def prepare_data(folder, categorical = False, test_size = 0.3):

  x, y, labels = basic_preparation(folder)
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=1)

  dataset = Dataset(x_train, y_train,x_test, y_test)
  dataset = preprocess(dataset, labels, categorical = categorical)
  return dataset