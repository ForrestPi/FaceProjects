import torch
import torch.nn as nn

from siriusbackbone import mobilenet_v2


class QualityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = mobilenet_v2()
        self.classify = nn.Linear(1280, 1)
    
    def forward(self, x):
        x = self.backbone(x)[0]
        x = torch.mean(x, dim=(2, 3))
        x = self.classify(x)
        return x