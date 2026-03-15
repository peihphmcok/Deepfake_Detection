import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init

pretrained_settings = {
    'xceptionnet_paper': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000
        }
    }
}

class Attention(nn.Module):
    """Lớp Attention để tăng cường tập trung vào các vùng quan trọng của hình ảnh."""
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 8, in_channels),
            nn.Sigmoid()
        )
        # Init weights
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SeparableConv2d(nn.Module):
    """Lớp Conv2d tách biệt cho kiến trúc Xception."""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        # Init weights
        init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.pointwise.weight, mode='fan_out', nonlinearity='relu')
        if self.conv1.bias is not None:
            init.constant_(self.conv1.bias, 0)
        if self.pointwise.bias is not None:
            init.constant_(self.pointwise.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class Block(nn.Module):
    """Khối cơ bản của Xception với kết nối dư."""
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
            init.kaiming_normal_(self.skip.weight, mode='fan_out', nonlinearity='relu')
        else:
            self.skip = None
        self.relu = nn.ReLU(inplace=True)
        rep = []
        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters
        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)
        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skipbn(self.skip(inp))
        else:
            skip = inp
        x += skip
        return x

class ImprovedXception(nn.Module):
    """Mô hình Xception cải tiến với cơ chế attention và dropout."""
    def __init__(self, num_classes=1000):
        super(ImprovedXception, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.attention1 = Attention(128)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.attention2 = Attention(256)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)
        self.attention3 = Attention(728)
        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.dropout = nn.Dropout(0.5)
        self.last_linear = nn.Linear(2048, num_classes)
        # Init weights
        init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.last_linear.weight, mode='fan_out', nonlinearity='relu')
        if self.last_linear.bias is not None:
            init.constant_(self.last_linear.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.attention1(x)
        x = self.block2(x)
        x = self.attention2(x)
        x = self.block3(x)
        x = self.attention3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.last_linear(x)
        return x

def improved_xception(num_classes=1000, pretrained='imagenet'):
    """
    Tạo mô hình ImprovedXception với tùy chọn tải trọng số pretrained.

    Args:
        num_classes (int): Số lớp đầu ra.
        pretrained (str): 'imagenet' để tải trọng số pretrained, None để khởi tạo ngẫu nhiên.

    Returns:
        ImprovedXception: Mô hình đã được khởi tạo.
    """
    if pretrained is not None:
        settings = pretrained_settings['xceptionnet_paper'][pretrained]
        model = ImprovedXception(num_classes=1000)
        state_dict = model_zoo.load_url(settings['url'])
        for k in list(state_dict.keys()):
            if 'pointwise.weight' in k and len(state_dict[k].shape) == 2:
                out_channels, in_channels = state_dict[k].shape
                state_dict[k] = state_dict[k].view(out_channels, in_channels, 1, 1)
        model.load_state_dict(state_dict, strict=False)
        model.last_linear = nn.Linear(model.last_linear.in_features, num_classes)
        init.kaiming_normal_(model.last_linear.weight, mode='fan_out', nonlinearity='relu')
        if model.last_linear.bias is not None:
            init.constant_(model.last_linear.bias, 0)
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    else:
        model = ImprovedXception(num_classes=num_classes)
    return model