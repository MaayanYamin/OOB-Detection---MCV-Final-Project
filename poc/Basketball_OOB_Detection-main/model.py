"""
Model architecture for OOB Detection
MobileNetV2 + LSTM with jersey color embedding
"""

import torch
import torch.nn as nn
from torchvision import models


class OOBModelWithColor(nn.Module):
    """Model that uses both video frames and jersey color metadata"""

    def __init__(self, num_classes=2):
        super().__init__()

        mobilenet = models.mobilenet_v2(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(mobilenet.children())[:-1])

        for param in list(self.feature_extractor.parameters())[:-10]:
            param.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=1280,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )

        self.color_embedding = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 + 16, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, frames, color):
        batch_size, time_steps = frames.size(0), frames.size(1)

        x = frames.view(-1, frames.size(2), frames.size(3), frames.size(4))
        features = self.feature_extractor(x)
        features = features.mean(dim=[2, 3])
        features = features.view(batch_size, time_steps, -1)

        lstm_out, _ = self.lstm(features)
        final_features = lstm_out[:, -1, :]

        color_features = self.color_embedding(color)

        combined = torch.cat([final_features, color_features], dim=1)
        output = self.classifier(combined)
        return output
