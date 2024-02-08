import torch
import torch.nn as nn
import torchvision.models as models



class VGGClassifier(nn.Module):
    def __init__(self, num_classes):
        super(VGGClassifier, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        self.vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        # Freeze the first half of VGG layers
        for param in self.vgg.features[:14].parameters():
            param.requires_grad = False

        self.vgg.classifier[-1] = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.vgg.features(x)
        x = self.vgg.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.vgg.classifier(x)
        return x

model = VGGClassifier(num_classes=2)
