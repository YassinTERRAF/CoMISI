
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalNetwork(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(MultimodalNetwork, self).__init__()
        # Dimension settings
        self.embed_size = 256
        
        # Audio branch
        self.audio_fc = nn.Linear(600, self.embed_size)
        # self.audio_dropout = nn.Dropout(dropout_rate)  # Dropout for audio branch
        
        # Visual branch
        self.visual_fc = nn.Linear(512, self.embed_size)
        # self.visual_dropout = nn.Dropout(dropout_rate)  # Dropout for visual branch
        
        # Combined features
        self.combined_fc1 = nn.Linear(self.embed_size * 2, self.embed_size)  # After concatenation
        self.combined_fc2 = nn.Linear(self.embed_size, 128)
        self.combined_dropout = nn.Dropout(dropout_rate)  # Additional dropout layer
        
        self.output_fc = nn.Linear(128, 24)  # Assuming 34 classes

    def forward(self, audio_features, visual_features):
        # Process audio features with dropout
        audio_out = F.relu(self.audio_fc(audio_features))
        
        # Process visual features with dropout
        visual_out = F.relu(self.visual_fc(visual_features))
        
        # Combine audio and visual features using simple concatenation
        combined = torch.cat((audio_out, visual_out), dim=1)
        
        # Further processing with dropout
        combined = F.relu(self.combined_fc1(combined))
        combined = self.combined_dropout(combined)
        combined = F.relu(self.combined_fc2(combined))
        
        # Final output
        output = self.output_fc(combined)
        return output