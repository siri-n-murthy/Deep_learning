"""
Compare two images using the trained Siamese network.

Usage:
    python compare.py path/to/image_a.png path/to/image_b.png
    python compare.py path/to/image_a.png path/to/image_b.png --threshold 0.7

Prints the Euclidean distance between their embeddings and a same/different
prediction based on a distance threshold.

NOTE on the threshold: the model was trained with contrastive loss using
margin=1.0, meaning "different" pairs are pushed to be at least ~1.0 apart
and "same" pairs pulled toward 0. So a threshold somewhere around 0.5-0.7
is a reasonable starting guess, but the right value depends on your data --
see find_threshold.py to calibrate it properly on your evaluation set
instead of guessing.
"""

import argparse

import torch
from PIL import Image

from model import SiameseNetwork
from data import TRANSFORM


def load_image(path: str):
    img = Image.open(path).convert("L")  # grayscale, matches training data
    return TRANSFORM(img).unsqueeze(0)   # add batch dim -> (1, 1, 105, 105)


def compare(model, path_a: str, path_b: str, device: str):
    img_a = load_image(path_a).to(device)
    img_b = load_image(path_b).to(device)

    model.eval()
    with torch.no_grad():
        emb_a = model.embed(img_a)
        emb_b = model.embed(img_b)
        distance = torch.nn.functional.pairwise_distance(emb_a, emb_b, p=2).item()

    return distance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_a", type=str)
    parser.add_argument("image_b", type=str)
    parser.add_argument("--model-path", type=str, default="siamese_model.pt")
    parser.add_argument("--threshold", type=float, default=0.6,
                         help="Distance below this = predicted 'same'. "
                              "Run find_threshold.py to calibrate this properly.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SiameseNetwork(embedding_dim=64).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    distance = compare(model, args.image_a, args.image_b, device)
    prediction = "SAME character" if distance < args.threshold else "DIFFERENT characters"

    print(f"Distance: {distance:.4f}")
    print(f"Threshold: {args.threshold}")
    print(f"Prediction: {prediction}")
