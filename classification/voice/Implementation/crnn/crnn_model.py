import torch
import torch.nn as nn
import torch.nn.functional as F


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduction = max(8, channel // 16)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DropBlock2D(nn.Module):
    def __init__(self, drop_prob=0.1, block_size=5):
        super(DropBlock2D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        gamma = self.drop_prob / (self.block_size ** 2)
        mask = (torch.rand(x.shape[0], *x.shape[2:], device=x.device) < gamma).float()
        mask = 1 - F.max_pool2d(mask.unsqueeze(1), kernel_size=self.block_size, stride=1, padding=self.block_size // 2)
        mask = mask.repeat(1, x.shape[1], 1, 1)
        return x * mask / (1 - self.drop_prob)


class FrequencyWeighting(nn.Module):
    def __init__(self, channels, freq_dim=299):
        super(FrequencyWeighting, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, freq_dim))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.weight)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, drop_prob=0.1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 7), stride=stride, padding=(1, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 7), stride=1, padding=(1, 3), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SELayer(out_channels)
        self.dropblock = DropBlock2D(drop_prob=drop_prob)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.dropblock(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x):
        attn_weights = F.softmax(self.attention(x), dim=1)
        return torch.sum(x * attn_weights, dim=1)


class ImprovedCRNN(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3):
        super(ImprovedCRNN, self).__init__()
        print("Initializing ImprovedCRNN")

        self.conv_layers = nn.Sequential(
            ResidualBlock(1, 40, stride=1, drop_prob=0.1),
            nn.MaxPool2d(2, 2),
            FrequencyWeighting(channels=40, freq_dim=149),
            ResidualBlock(40, 80, stride=1, drop_prob=0.1),
            nn.MaxPool2d(2, 2),
            ResidualBlock(80, 120, stride=1, drop_prob=0.1),
            nn.MaxPool2d(2, 2),
        )

        self.gru = nn.GRU(
            input_size=120 * 37,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.norm1 = nn.InstanceNorm1d(120 * 37, affine=True)
        self.norm2 = nn.LayerNorm(128 * 2)
        self.attention_pool = AttentionPooling(hidden_size=128 * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(128 * 2, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, x):
        features = self.conv_layers(x)
        batch_size, channels, time, freq = features.size()
        features = features.permute(0, 2, 1, 3).contiguous()
        features = features.view(batch_size, time, channels * freq)
        features = self.norm1(features.transpose(1, 2)).transpose(1, 2)
        features, _ = self.gru(features)
        features = self.norm2(features)
        features = self.dropout(features)
        output = self.attention_pool(features)
        output = self.classifier(output)
        return output