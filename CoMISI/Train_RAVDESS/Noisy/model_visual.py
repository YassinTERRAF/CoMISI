import torch
import torch.nn as nn
import torch.nn.functional as F

class VisualNetwork(nn.Module):
    def __init__(self, dropout_rate=0.2):  # Added dropout_rate parameter with a default of 0.5
        super(VisualNetwork, self).__init__()
        # Visual branch only
        self.visual_fc1 = nn.Linear(512, 256)  # Assuming visual feature dimension is 512
        self.visual_fc2 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(dropout_rate)  # Second dropout layer
        self.output_fc = nn.Linear(128, 24)  # Assuming 24 classes

    def forward(self, visual_features):
        visual_out = F.relu(self.visual_fc1(visual_features))
        visual_out = F.relu(self.visual_fc2(visual_out))
        visual_out = self.dropout1(visual_out)  # Apply dropout after second activation
        output = self.output_fc(visual_out)
        return output
