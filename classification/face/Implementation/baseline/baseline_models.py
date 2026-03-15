import torch
import torch.nn as nn
import timm
import torchvision.models as models

class XceptionBaseline(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.5):
        super(XceptionBaseline, self).__init__()
        self.model = timm.create_model('xception', pretrained=True, num_classes=num_classes)
        self.classifier_name = 'fc'
    def forward(self, x):
        return self.model(x)

class MobileNetV3Baseline(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.5):
        super(MobileNetV3Baseline, self).__init__()
        self.model = timm.create_model('mobilenetv3_large_100', pretrained=True)
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, num_classes)
        self.classifier_name = 'classifier'
    def forward(self, x):
        return self.model(x)

class EfficientNetB0Baseline(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.5):
        super(EfficientNetB0Baseline, self).__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True)
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, num_classes)
        self.classifier_name = 'classifier'
    def forward(self, x):
        return self.model(x)

class ResNet18Baseline(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.5):
        super(ResNet18Baseline, self).__init__()
        self.model = models.resnet18(weights='IMAGENET1K_V1')
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        self.classifier_name = 'fc'
    def forward(self, x):
        return self.model(x)

class ShuffleNetV2Baseline(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.5):
        super(ShuffleNetV2Baseline, self).__init__()
        self.model = models.shufflenet_v2_x1_0(weights='IMAGENET1K_V1')
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        self.classifier_name = 'fc'
    def forward(self, x):
        return self.model(x)

def create_xception_baseline(dropout_rate=0.5):
    return XceptionBaseline(num_classes=1, dropout_rate=dropout_rate)

def create_mobilenetv3_baseline(dropout_rate=0.5):
    return MobileNetV3Baseline(num_classes=1, dropout_rate=dropout_rate)

def create_efficientnet_b0_baseline(dropout_rate=0.5):
    return EfficientNetB0Baseline(num_classes=1, dropout_rate=dropout_rate)

def create_resnet18_baseline(dropout_rate=0.5):
    return ResNet18Baseline(num_classes=1, dropout_rate=dropout_rate)

def create_shufflenet_v2_baseline(dropout_rate=0.5):
    return ShuffleNetV2Baseline(num_classes=1, dropout_rate=dropout_rate)

BASELINE_MODELS = {
    'mobilenetv3_large': {
        'model_fn': create_mobilenetv3_baseline,
        'name': 'MobileNetV3-Large',
        'description': 'Original MobileNetV3-Large architecture',
        'normalization': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        'classifier_name': 'classifier'
    },
    'resnet18': {
        'model_fn': create_resnet18_baseline,
        'name': 'ResNet18',
        'description': 'Original ResNet18 architecture',
        'normalization': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        'classifier_name': 'fc'
    },
    'shufflenet_v2': {
        'model_fn': create_shufflenet_v2_baseline,
        'name': 'ShuffleNet V2',
        'description': 'Original ShuffleNet V2 architecture',
        'normalization': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        'classifier_name': 'fc'
    },

    'efficientnet_b0': {
        'model_fn': create_efficientnet_b0_baseline,
        'name': 'EfficientNet-B0',
        'description': 'Original EfficientNet-B0 architecture',
        'normalization': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        'classifier_name': 'classifier'
    },

    'xception': {
        'model_fn': create_xception_baseline,
        'name': 'Xception',
        'description': 'Original Xception architecture',
        'normalization': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        'classifier_name': 'fc'
    },
}