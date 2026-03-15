import torch
import torch.nn as nn
from torchvision.models import squeezenet1_1, shufflenet_v2_x0_5, efficientnet_b0

class SqueezeNetBaseline(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.4):
        super(SqueezeNetBaseline, self).__init__()
        self.model = squeezenet1_1(pretrained=True)
        self.model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
        self.model.classifier[2] = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        return self.model(x)

class ShuffleNetBaseline(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.4):
        super(ShuffleNetBaseline, self).__init__()
        self.model = shufflenet_v2_x0_5(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

class EfficientNetB0Baseline(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.4):
        super(EfficientNetB0Baseline, self).__init__()
        self.model = efficientnet_b0(pretrained=True)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)