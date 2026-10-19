import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from src.dataset import LABELS


def create_model(pretrained=True):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    # Replace the original 1000-class ImageNet head with 14 disease outputs.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(LABELS))

    return model