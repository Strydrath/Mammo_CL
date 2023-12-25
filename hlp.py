import torch
import torch

'''
base_model = applications.VGG16(
      include_top=False,
      weights="imagenet",
      input_shape=(512,512,1)
  )

  base_model.trainable = True
  num_layers = len(base_model.layers)
  num_layers_to_freeze = int(num_layers*0.8)

  for layer in base_model.layers[:num_layers_to_freeze]:
      layer.trainable = False

  model_2 = tf.keras.models.Sequential([
      base_model,
      layers.GlobalAveragePooling2D(),
      layers.Dropout(0.2),
      layers.Dense(128),
      layers.Dense(1, activation = 'sigmoid'),
    ]
  )
  opt = tf.keras.optimizers.Adam(learning_rate=0.01)
  model_2.compile(optimizer=opt, loss="BinaryCrossentropy", metrics=["accuracy"])
  callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)
  checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    "./model",
    monitor = "val_loss",
    save_best_only = True,
    )
'''

import torch.nn as nn
import torch.optim as optim

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.base_model = torch.hub.load('pytorch/vision:v0.9.0', 'vgg16', pretrained=True)
        self.base_model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.base_model.classifier[6] = nn.Linear(4096, 1)
        
    def forward(self, x):
        x = self.base_model(x)
        return x

model = CustomModel()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

# Define your dataset and dataloader here

num_layers = len(model.base_model.features)
num_layers_to_freeze = int(num_layers * 0.8)

for i, param in enumerate(model.base_model.features.parameters()):
    if i < num_layers_to_freeze:
        param.requires_grad = False

# Train your model here

# Save the best model
best_model_path = "./model.pth"
torch.save(model.state_dict(), best_model_path)


