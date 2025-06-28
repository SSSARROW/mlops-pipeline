import logging
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
class Model(ABC):
    """Abstract class for all models"""

    @abstractmethod
    def train(self,X_train, y_train):
        """Train the model with the provided training data.
        Args:
            X_train : Training data.
            y_train : Training labels.
        """

        pass

class RainNet(nn.Module):
    def __init__(self, input_size):
        super(RainNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.model(x)
    
