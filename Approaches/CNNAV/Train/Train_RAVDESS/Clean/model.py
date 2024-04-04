
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalNetwork(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(MultimodalNetwork, self).__init__()


        # Updated dimension settings for common size
        self.common_embed_size = 512
        
        # Audio branch
        # Keep audio feature input size as 192, but transform to common size of 512
        self.audio_fc = nn.Linear(192, self.common_embed_size)
        
        # Visual branch
        # Update visual feature input size to 2622, transform to common size of 512
        self.visual_fc = nn.Linear(2622, self.common_embed_size)
        
        # Combined features
        # Now, as both features are of size 512, the combined size is 1024
        self.combined_fc1 = nn.Linear(self.common_embed_size * 2, self.common_embed_size)  # Reduce dimension from 1024 to 512
        self.combined_dropout = nn.Dropout(dropout_rate)  # Dropout layer before the final classification layer
        
        self.output_fc = nn.Linear(self.common_embed_size, 24)  # Assuming 34 is the number of classes for classification

    def forward(self, audio_features, visual_features):
        # Process audio features
        audio_out = F.relu(self.audio_fc(audio_features))
        
        # Process visual features
        visual_out = F.relu(self.visual_fc(visual_features))
        
        # Combine audio and visual features using simple concatenation
        combined = torch.cat((audio_out, visual_out), dim=1)
        
        # Further processing and dimension reduction
        combined = F.relu(self.combined_fc1(combined))
        
        # Apply dropout before the final classification layer
        combined = self.combined_dropout(combined)
        
        # Final output for classification
        output = self.output_fc(combined)
        return output