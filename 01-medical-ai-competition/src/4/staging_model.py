import torch.nn as nn
import torchvision.models as models
class LesionStagingClassifier(nn.Module):
    def __init__(self, num_features=7):
        super().__init__()
        backbone = models.resnet50(weights='IMAGENET1K_V2')
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool
        )
        self.fc = nn.Linear(2048, num_features)
    def forward(self, x):
        feat = self.features(x)
        feat = feat.flatten(1)
        return self.fc(feat)
