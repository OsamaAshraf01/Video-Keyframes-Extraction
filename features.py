import torch
from abc import ABC, abstractmethod
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

class FeatureExtractor(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def preprocess(self, img:torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def extract_features(self, img:torch.Tensor) -> torch.Tensor:
        pass

class ResNetFeatureExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__()
        self.weights = ResNet50_Weights.IMAGENET1K_V2
        self.resnet = resnet50(weights=self.weights)
        self.resnet.fc = nn.Identity() # remove final FC layer
        self.resnet.eval()

    def preprocess(self, img:torch.Tensor) -> torch.Tensor:
        img = img.to(self.resnet.conv1.weight.device)
        return self.weights.transforms()(img)

    def extract_features(self, img:torch.Tensor) -> torch.Tensor:
        return self.resnet(img)

    def train(self, mode=True):
        super().train(mode)
        self.resnet.eval()
        return self
