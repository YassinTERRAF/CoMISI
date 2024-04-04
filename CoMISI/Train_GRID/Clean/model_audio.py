import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioNetwork(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(AudioNetwork, self).__init__()
        # Audio branch only
        self.audio_fc1 = nn.Linear(192, 128)  # Input dimension to first layer
        self.dropout1 = nn.Dropout(dropout_rate)  # Second dropout layer

        self.output_fc = nn.Linear(128, 34)  # Output layer, assuming 24 classes based on your previous output layer

    def forward(self, audio_features):
        audio_out = F.relu(self.audio_fc1(audio_features))
        audio_out = self.dropout1(audio_out)  # Apply dropout after second activation

        output = self.output_fc(audio_out)
        return output