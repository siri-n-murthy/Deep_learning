"""
Calibrates the distance threshold used by compare.py, instead of guessing one.

Runs the model over many labeled pairs from the evaluation set, then finds
the threshold that maximizes verification accuracy (same-vs-different).
This is also a legitimate metric to report in your write-up alongside
N-way one-shot accuracy.

Usage:
    python find_threshold.py --num-pairs 1000
"""

import argparse

import torch
import numpy as np

from data import OmniglotPairDataset
from model import SiameseNetwork


def find_best_threshold(model, dataset, device: str):
    model.eval()
    distances = []
    labels = []

    with torch.no_grad():
        for i in range(len(dataset)):
            img_a, img_b, label = dataset[i]
            img_a, img_b = img_a.unsqueeze(0).to(device), img_b.unsqueeze(0).to(device)

            emb_a = model.embed(img_a)
            emb_b = model.embed(img_b)
            d = torch.nn.functional.pairwise_distance(emb_a, emb_b, p=2).item()

            distances.append(d)
            labels.append(label.item())  # 0 = same, 1 = different

    distances = np.array(distances)
    labels = np.array(labels)

    best_threshold, best_acc = 0.0, 0.0
    for t in np.arange(0.0, distances.max(), 0.01):
        # predict "same" (0) if distance < t, "different" (1) otherwise
        predictions = (distances >= t).astype(float)
        acc = (predictions == labels).mean()
        if acc > best_acc:
            best_acc, best_threshold = acc, t

    return best_threshold, best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-pairs", type=int, default=1000)
    parser.add_argument("--model-path", type=str, default="siamese_model.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SiameseNetwork(embedding_dim=64).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    eval_dataset = OmniglotPairDataset(background=False, num_pairs=args.num_pairs)

    best_threshold, best_acc = find_best_threshold(model, eval_dataset, device)
    print(f"Best threshold: {best_threshold:.3f}")
    print(f"Verification accuracy at that threshold: {best_acc * 100:.2f}%")
    print(f"\nUse this in compare.py with: --threshold {best_threshold:.3f}")
