import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, hidden_dim):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(value_dim, hidden_dim)

    def forward(self, query, key, value):
        query = self.query_proj(query.unsqueeze(1))
        key = self.key_proj(key.unsqueeze(1))
        value = self.value_proj(value.unsqueeze(1))

        # Calculate scores and apply softmax
        scores = torch.bmm(query, key.transpose(1, 2))
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention weights to value
        attended = torch.bmm(attn_weights, value).squeeze(1)

        return attended

class MultimodalNetworkAttention(nn.Module):
    def __init__(self, audio_dim=192, visual_dim=512, shared_dim=128, output_dim=34, dropout_rate=0.5):
        super().__init__()
        
        # Initialize cross-attention modules for both directions
        self.audio_to_visual_attention = CrossAttention(audio_dim, visual_dim, visual_dim, shared_dim)
        self.visual_to_audio_attention = CrossAttention(visual_dim, audio_dim, audio_dim, shared_dim)
        
        # Projection layers to shared dimension
        self.audio_proj = nn.Linear(shared_dim, shared_dim)
        self.visual_proj = nn.Linear(shared_dim, shared_dim)

        # Define layers for combining and classifying
        self.combined_fc = nn.Linear(2 * shared_dim, shared_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.output_fc = nn.Linear(shared_dim, output_dim)

    def forward(self, audio_features, visual_features):
        # Apply cross-modal attention
        audio_attended = self.visual_to_audio_attention(visual_features, audio_features, audio_features)
        visual_attended = self.audio_to_visual_attention(audio_features, visual_features, visual_features)

        # Project attended features to shared dimension
        audio_proj = F.relu(self.audio_proj(audio_attended))
        visual_proj = F.relu(self.visual_proj(visual_attended))

        # Combine audio and visual features
        combined_features = torch.cat((audio_proj, visual_proj), dim=1)
        combined_features = self.dropout(F.relu(self.combined_fc(combined_features)))

        # Classify combined features
        output = self.output_fc(combined_features)

        return output
