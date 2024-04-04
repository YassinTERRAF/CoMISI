import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftFeatureEmphasisModule(nn.Module):
    def __init__(self, feature_size=128):
        super(SoftFeatureEmphasisModule, self).__init__()
        self.importance_scores = nn.Parameter(torch.ones(feature_size))

    def forward(self, features):
        emphasized_features = features * self.importance_scores
        return emphasized_features

class EnhancedCMIFModule(nn.Module):
    def __init__(self, audio_size, visual_size, hidden_size=128):
        super(EnhancedCMIFModule, self).__init__()
        self.audio_proj = nn.Linear(audio_size, hidden_size)
        self.visual_proj = nn.Linear(visual_size, hidden_size)
        self.audio_bn = nn.BatchNorm1d(hidden_size)
        self.visual_bn = nn.BatchNorm1d(hidden_size)
        self.audio_emphasis = SoftFeatureEmphasisModule(hidden_size)
        self.visual_emphasis = SoftFeatureEmphasisModule(hidden_size)
        self.relation_distill = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, audio_features, visual_features, return_interactions=False):
        audio_hidden = self.relu(self.audio_bn(self.audio_proj(audio_features)))
        visual_hidden = self.relu(self.visual_bn(self.visual_proj(visual_features)))
        audio_emphasized = self.audio_emphasis(audio_hidden)
        visual_emphasized = self.visual_emphasis(visual_hidden)
        interaction = audio_emphasized * visual_emphasized
        relation = self.relation_distill(interaction)
        enhanced_audio = audio_emphasized + relation
        enhanced_visual = visual_emphasized + relation
        
        if return_interactions:
            return enhanced_audio, enhanced_visual, interaction, relation
        return enhanced_audio, enhanced_visual

class MultimodalNetworkCoMISI(nn.Module):
    def __init__(self, hidden_size=128, dropout_rate=0.5):
        super(MultimodalNetworkCoMISI, self).__init__()
        self.audio_fc = nn.Linear(192, hidden_size)
        self.visual_fc = nn.Linear(512, hidden_size)
        self.audio_dropout = nn.Dropout(dropout_rate)
        self.visual_dropout = nn.Dropout(dropout_rate)
        self.cmr_module = EnhancedCMIFModule(192, 512, hidden_size)
        self.fusion_fc = nn.Linear(hidden_size * 2, hidden_size)
        self.fusion_dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, 24)

    def forward(self, audio_features, visual_features, return_interactions=False):
        audio_features = self.audio_dropout(audio_features)
        visual_features = self.visual_dropout(visual_features)
        if return_interactions:
            enhanced_audio, enhanced_visual, interaction, relation = self.cmr_module(audio_features, visual_features, return_interactions=True)
            fused_features = torch.cat([enhanced_audio, enhanced_visual], dim=1)
            fused_features = self.fusion_fc(fused_features)
            fused_features = self.fusion_dropout(fused_features)
            output = self.classifier(fused_features)
            return output, interaction, relation
        else:
            enhanced_audio, enhanced_visual = self.cmr_module(audio_features, visual_features)
            fused_features = torch.cat([enhanced_audio, enhanced_visual], dim=1)
            fused_features = self.fusion_fc(fused_features)
            fused_features = self.fusion_dropout(fused_features)
            output = self.classifier(fused_features)
            return output
