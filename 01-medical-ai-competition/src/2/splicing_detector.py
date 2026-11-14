import torch
import torch.nn as nn
import torchvision.models as models
class SplicingDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        self.backbone.fc = nn.Linear(512, 2)
    def forward(self, x):
        return self.backbone(x)
