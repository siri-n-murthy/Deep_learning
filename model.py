"""
Siamese network: a single small CNN "tower" that both images are passed
through (shared weights), producing an embedding vector for each. Distance
between embeddings is computed by the loss function / eval code, not here.
"""

import torch
import torch.nn as nn


class EmbeddingNet(nn.Module):
    """The shared CNN tower. Input: (B, 1, 105, 105) -> Output: (B, embedding_dim)."""

    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=10), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=7), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=4), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=4), nn.ReLU(inplace=True),
        )
        # figure out flattened size dynamically so we don't hardcode it
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 105, 105)
            flat_size = self.conv(dummy).view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flat_size, 512), nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SiameseNetwork(nn.Module):
    """
    Wraps EmbeddingNet and runs it on both inputs with SHARED weights
    (it's literally the same module called twice, not two separate copies).
    """

    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.embedding_net = EmbeddingNet(embedding_dim)

    def forward(self, img_a, img_b):
        emb_a = self.embedding_net(img_a)
        emb_b = self.embedding_net(img_b)
        return emb_a, emb_b

    def embed(self, x):
        """Useful at inference time when you just need one embedding."""
        return self.embedding_net(x)
