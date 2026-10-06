import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.datasets import Omniglot

from data import TRANSFORM


def _index_by_class(dataset):
    """Map class_label -> list of indices into `dataset` belonging to that class."""
    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset._flat_character_images):
        class_to_indices[label].append(idx)
    return class_to_indices


def build_reference_gallery(background: bool = False, max_classes: int = 20, transform=TRANSFORM):
    """Build a small gallery of one example per Omniglot class for nearest-neighbor recognition."""
    dataset = Omniglot(root="./omniglot_data", background=background, download=True, transform=transform)
    class_to_indices = _index_by_class(dataset)
    selected_classes = sorted(class_to_indices.keys())[:max_classes]

    if not selected_classes:
        raise ValueError("No Omniglot classes were found in the selected split.")

    gallery = []
    labels = []
    for cls in selected_classes:
        idx = class_to_indices[cls][0]
        img, _ = dataset[idx]
        gallery.append(img)
        labels.append(str(cls))

    return torch.stack(gallery), labels


def predict_single_character(model, query_image, device: str = "cpu", background: bool = False, max_classes: int = 20):
    """Classify one image by finding the nearest reference prototype in embedding space."""
    gallery, labels = build_reference_gallery(background=background, max_classes=max_classes)
    query_tensor = TRANSFORM(query_image.convert("L")).unsqueeze(0).to(device)

    with torch.no_grad():
        gallery = gallery.to(device)
        reference_emb = model.embed(gallery)
        query_emb = model.embed(query_tensor)
        distances = F.pairwise_distance(reference_emb, query_emb.expand_as(reference_emb), p=2)
        best_idx = int(torch.argmin(distances).item())

    return labels[best_idx], float(distances[best_idx].item()), labels, distances.cpu()


def main():
    parser = argparse.ArgumentParser(description="Recognize a single handwritten character using the trained Siamese model.")
    parser.add_argument("--image", type=str, required=True, help="Path to the query image to classify.")
    parser.add_argument("--model-path", type=str, default="siamese_model.pt", help="Path to the trained model weights.")
    parser.add_argument("--max-classes", type=int, default=20, help="Number of reference classes to compare against.")
    parser.add_argument("--background", action="store_true", help="Use the training split as the gallery; default uses the test split.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from model import SiameseNetwork

    model = SiameseNetwork(embedding_dim=64).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    query_image = Image.open(args.image).convert("L")
    label, distance, _, _ = predict_single_character(
        model,
        query_image,
        device=device,
        background=args.background,
        max_classes=args.max_classes,
    )

    print(f"Predicted class: {label}")
    print(f"Nearest distance: {distance:.4f}")


if __name__ == "__main__":
    main()
