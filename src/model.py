import torch
from torch import nn
from torchvision import models


class FaceModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
    ):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.model = models.resnet50(weights=weights)
        self.model.fc = nn.Linear(in_features=2048, out_features=num_classes)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    net = FaceModel(num_classes=7)
    # dummy data (Batch Size 1, 3 Channels, 224x224)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = net(dummy_input)
    print(f"Model Output Shape: {output.shape}")
    # Expected: torch.Size([1, 7])
