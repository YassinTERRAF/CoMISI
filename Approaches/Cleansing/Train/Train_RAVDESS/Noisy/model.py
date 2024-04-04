import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalNetwork(nn.Module):
    def __init__(self, num_classes=24, dropout_rate=0.5):
        super(MultimodalNetwork, self).__init__()

        # Audio branch: Transform audio input size 192 to 1024
        self.audio_fc = nn.Linear(192, 1024)
        
        # No transformation is applied to the visual feature as it's already of size 512

        # Direct classification layer after concatenation of transformed audio (1024) and visual (512) features
        self.output_fc = nn.Linear(1024 + 512, num_classes)  # Input size is 1536 for classification
        
        # Dropout layer for regularization before classification
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, audio_features, visual_features):
        # Process audio features through transformation to 1024
        audio_out = F.relu(self.audio_fc(audio_features))
        
        # Visual features are kept as is, since they're already correctly sized
        
        # Combine audio and visual features using simple concatenation
        combined = torch.cat((audio_out, visual_features), dim=1)
        
        # Apply dropout before the final classification
        combined = self.dropout(combined)
        
        # Final output for classification
        output = self.output_fc(combined)
        return output