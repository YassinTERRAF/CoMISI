# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalNetworkWithScoreFusion(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(MultimodalNetworkWithScoreFusion, self).__init__()
        self.embed_size = 128

        # Audio branch
        self.audio_fc1 = nn.Linear(192, self.embed_size)
        self.audio_fc2 = nn.Linear(self.embed_size, 24)  # Assuming 24 classes

        # Visual branch
        self.visual_fc1 = nn.Linear(512, self.embed_size)
        self.dropout1 = nn.Dropout(dropout_rate)  # Second dropout layer

        self.visual_fc2 = nn.Linear(self.embed_size, 24)  # Assuming 24 classes
        

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
        fused_scores = (audio_scores + visual_scores) 


        return fused_scores
