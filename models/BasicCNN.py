
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Rescaling, RandomFlip, RandomRotation

# Define the model architecture
def create_model(im_height, im_width, num_classes):
    model = Sequential()
    data_augmentation = Sequential(
          [
              RandomFlip("horizontal",
                                input_shape=(im_height,
                                            im_width,
                                            1)),
              RandomRotation(0.2),
          ]
      )
    data_normalization = Rescaling(1. / 255)
    model.add(data_augmentation)
    model.add(data_normalization)
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, kernel_initializer="glorot_uniform", activation='softmax'))
    return model
