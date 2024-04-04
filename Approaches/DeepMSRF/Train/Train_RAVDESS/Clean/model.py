import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalNetwork(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(MultimodalNetwork, self).__init__()
        self.embed_size = 512  # Set embedding size to 512 as per new requirements

        # Audio branch: Adjust to receive embeddings of size 4096
        self.audio_fc1 = nn.Linear(4096, self.embed_size)
        self.audio_fc2 = nn.Linear(self.embed_size, 24)  # Assuming 34 classes

        # Visual branch: Adjust to receive embeddings of size 4096
        self.visual_fc1 = nn.Linear(4096, self.embed_size)
        self.dropout1 = nn.Dropout(dropout_rate)  # Use for regularization in both branches

        self.visual_fc2 = nn.Linear(self.embed_size, 24)  # Assuming 34 classes

    def forward(self, audio_features, visual_features):
        # Audio path
        audio_out = F.relu(self.audio_fc1(audio_features))
        audio_out = self.dropout1(audio_out)
        audio_scores = self.audio_fc2(audio_out)

        # Visual path
        visual_out = F.relu(self.visual_fc1(visual_features))
        visual_out = self.dropout1(visual_out)
        visual_scores = self.visual_fc2(visual_out)

        # Score-level fusion
        fused_scores = audio_scores + visual_scores  # Combine scores from both modalities

        return fused_scores
