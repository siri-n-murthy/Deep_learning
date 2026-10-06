"""
Contrastive loss (Hadsell, Chopra, LeCun 2006):

    L = (1 - Y) * 0.5 * D^2 + Y * 0.5 * max(0, margin - D)^2

Where:
    D      = Euclidean distance between the two embeddings
    Y      = 0 if same class, 1 if different class
    margin = how far apart "different" pairs should be pushed at minimum

Intuition:
    - Same-class pairs (Y=0): loss = 0.5 * D^2  -> directly penalizes any distance, pulls them together.
    - Different-class pairs (Y=1): loss = 0.5 * max(0, margin - D)^2 -> only penalizes if they're
      closer than `margin`. Once they're far enough apart, loss is 0 (no need to push infinitely far).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb_a, emb_b, label):
        distance = F.pairwise_distance(emb_a, emb_b, p=2)

        same_class_loss = (1 - label) * torch.pow(distance, 2)
        diff_class_loss = label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)

        loss = 0.5 * (same_class_loss + diff_class_loss)
        return loss.mean()
