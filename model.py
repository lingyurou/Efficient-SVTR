import torch
import torch.nn as nn
import torch.nn.functional as F

class JumbleModule(nn.Module):
    """
    Jumble Module: Introduces random spatial perturbations to enhance robustness.
    """
    def __init__(self, p=0.5):
        super(JumbleModule, self).__init__()
        self.p = p

    def forward(self, x):
        if self.training and torch.rand(1) < self.p:
            # Randomly shuffle spatial dimensions
            batch_size, channels, height, width = x.size()
            x = x.view(batch_size, channels, -1)
            idx = torch.randperm(x.size(2))
            x = x[:, :, idx].view(batch_size, channels, height, width)
        return x

class SelfDistillationModule(nn.Module):
    """
    Self-Distillation Module: Refines feature representations and compresses the model.
    """
    def __init__(self, in_channels, out_channels):
        super(SelfDistillationModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class SVTRCompact(nn.Module):
    """
    SVTR-Compact: Lightweight Scene Text Recognition model with Jumble and Self-Distillation modules.
    """
    def __init__(self, num_classes):
        super(SVTRCompact, self).__init__()
        self.jumble = JumbleModule(p=0.5)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.distill = SelfDistillationModule(64, 32)
        self.fc = nn.Linear(32 * 16 * 16, num_classes)  # Adjust dimensions as needed

    def forward(self, x):
        x = self.jumble(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.distill(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
