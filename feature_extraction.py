import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureExtractionException(Exception):
    """Base exception class for feature extraction module"""
    pass

class InvalidInputException(FeatureExtractionException):
    """Exception raised when invalid input is provided"""
    pass

class FeatureExtractionLayer(nn.Module):
    """
    Base class for feature extraction layers

    Attributes:
        input_dim (int): Input dimension
        output_dim (int): Output dimension
    """
    def __init__(self, input_dim: int, output_dim: int):
        super(FeatureExtractionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward method")

class Conv2DLayer(FeatureExtractionLayer):
    """
    Conv2D layer for feature extraction

    Attributes:
        input_dim (int): Input dimension
        output_dim (int): Output dimension
        kernel_size (int): Kernel size
        stride (int): Stride
        padding (int): Padding
    """
    def __init__(self, input_dim: int, output_dim: int, kernel_size: int, stride: int, padding: int):
        super(Conv2DLayer, self).__init__(input_dim, output_dim)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv2d = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2d(x)

class MaxPool2DLayer(FeatureExtractionLayer):
    """
    MaxPool2D layer for feature extraction

    Attributes:
        input_dim (int): Input dimension
        output_dim (int): Output dimension
        kernel_size (int): Kernel size
        stride (int): Stride
        padding (int): Padding
    """
    def __init__(self, input_dim: int, output_dim: int, kernel_size: int, stride: int, padding: int):
        super(MaxPool2DLayer, self).__init__(input_dim, output_dim)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.maxpool2d = nn.MaxPool2d(kernel_size, stride, padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool2d(x)

class FlattenLayer(FeatureExtractionLayer):
    """
    Flatten layer for feature extraction

    Attributes:
        input_dim (int): Input dimension
        output_dim (int): Output dimension
    """
    def __init__(self, input_dim: int, output_dim: int):
        super(FlattenLayer, self).__init__(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(-1, self.output_dim)

class DenseLayer(FeatureExtractionLayer):
    """
    Dense layer for feature extraction

    Attributes:
        input_dim (int): Input dimension
        output_dim (int): Output dimension
    """
    def __init__(self, input_dim: int, output_dim: int):
        super(DenseLayer, self).__init__(input_dim, output_dim)
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense(x)

class FeatureExtractor(nn.Module):
    """
    Feature extractor class

    Attributes:
        layers (List[FeatureExtractionLayer]): List of feature extraction layers
    """
    def __init__(self, layers: List[FeatureExtractionLayer]):
        super(FeatureExtractor, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

class FeatureExtractionConfig:
    """
    Feature extraction configuration class

    Attributes:
        input_dim (int): Input dimension
        output_dim (int): Output dimension
        kernel_size (int): Kernel size
        stride (int): Stride
        padding (int): Padding
    """
    def __init__(self, input_dim: int, output_dim: int, kernel_size: int, stride: int, padding: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

def create_feature_extractor(config: FeatureExtractionConfig) -> FeatureExtractor:
    """
    Create feature extractor based on configuration

    Args:
        config (FeatureExtractionConfig): Feature extraction configuration

    Returns:
        FeatureExtractor: Feature extractor instance
    """
    layers = [
        Conv2DLayer(config.input_dim, 64, config.kernel_size, config.stride, config.padding),
        MaxPool2DLayer(64, 128, config.kernel_size, config.stride, config.padding),
        FlattenLayer(128, 128),
        DenseLayer(128, config.output_dim)
    ]
    return FeatureExtractor(layers)

def validate_input(x: torch.Tensor, input_dim: int) -> None:
    """
    Validate input tensor

    Args:
        x (torch.Tensor): Input tensor
        input_dim (int): Expected input dimension

    Raises:
        InvalidInputException: If input is invalid
    """
    if x.dim() != 4:
        raise InvalidInputException("Input must be a 4D tensor")
    if x.shape[1] != input_dim:
        raise InvalidInputException("Input dimension mismatch")

def extract_features(x: torch.Tensor, feature_extractor: FeatureExtractor) -> torch.Tensor:
    """
    Extract features from input tensor

    Args:
        x (torch.Tensor): Input tensor
        feature_extractor (FeatureExtractor): Feature extractor instance

    Returns:
        torch.Tensor: Extracted features
    """
    validate_input(x, feature_extractor.layers[0].input_dim)
    return feature_extractor(x)

def main():
    # Create feature extraction configuration
    config = FeatureExtractionConfig(3, 10, 3, 2, 1)

    # Create feature extractor
    feature_extractor = create_feature_extractor(config)

    # Create input tensor
    x = torch.randn(1, 3, 224, 224)

    # Extract features
    features = extract_features(x, feature_extractor)

    # Print extracted features
    print(features)

if __name__ == "__main__":
    main()