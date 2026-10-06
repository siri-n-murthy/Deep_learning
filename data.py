"""
Omniglot data loading + pair generation for Siamese network training.

Uses torchvision's built-in Omniglot dataset, which already gives us the
standard split used in few-shot learning papers:
  - background=True  -> 964 character classes, used for TRAINING
  - background=False -> 659 character classes, used for TESTING (one-shot eval)

Critically, the alphabets/classes in these two sets do NOT overlap. This is
what makes the evaluation "few-shot" / "one-shot" -- the model has never
seen these characters during training, and we test whether its learned
embedding space still lets it tell them apart.
"""

import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from torchvision.datasets import Omniglot
from torchvision import transforms


IMAGE_SIZE = 105  # native Omniglot resolution

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),  # -> [0, 1], shape (1, 105, 105), already grayscale
])


def _index_by_class(dataset):
    """Map class_label -> list of indices into `dataset` belonging to that class."""
    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset._flat_character_images):
        class_to_indices[label].append(idx)
    return class_to_indices


class OmniglotPairDataset(Dataset):
    """
    Wraps torchvision's Omniglot dataset and serves random (img_a, img_b, label)
    pairs, where label = 0 if same character class, 1 if different.

    We follow the common convention here (0=same, 1=different) so it lines up
    directly with the standard contrastive loss formula:
        L = (1-Y) * 0.5 * D^2 + Y * 0.5 * max(0, margin - D)^2
    i.e. Y=0 pulls same-class pairs together, Y=1 pushes different-class pairs apart.

    `num_pairs` pairs are (re)sampled fresh each epoch via a large virtual length --
    with ~1000 classes and only ~20 images each, sampling on-the-fly gives far more
    effective variety than pre-generating a fixed pair list.
    """

    def __init__(self, background: bool, num_pairs: int = 30000, transform=TRANSFORM):
        self.base = Omniglot(root="./omniglot_data", background=background,
                              download=True, transform=transform)
        self.class_to_indices = _index_by_class(self.base)
        self.classes = list(self.class_to_indices.keys())
        self.num_pairs = num_pairs

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        # 50/50 split between same-class and different-class pairs
        same_class = random.random() < 0.5

        if same_class:
            cls = random.choice(self.classes)
            idx_a, idx_b = random.sample(self.class_to_indices[cls], 2) \
                if len(self.class_to_indices[cls]) >= 2 \
                else (self.class_to_indices[cls][0], self.class_to_indices[cls][0])
            label = 0.0
        else:
            cls_a, cls_b = random.sample(self.classes, 2)
            idx_a = random.choice(self.class_to_indices[cls_a])
            idx_b = random.choice(self.class_to_indices[cls_b])
            label = 1.0

        img_a, _ = self.base[idx_a]
        img_b, _ = self.base[idx_b]

        return img_a, img_b, torch.tensor(label, dtype=torch.float32)


class NWayOneShotTask:
    """
    Builds N-way one-shot classification episodes from the evaluation
    (unseen-during-training) split, for a more meaningful metric than
    raw pair accuracy.

    One episode:
      - Pick N classes at random.
      - Pick one "support" image from each of the N classes.
      - Pick one "query" image from ONE of those N classes (a different
        sample than its support image).
      - Task: match the query to the correct support image, using the
        learned embedding distance. Model has never seen these classes.
    """

    def __init__(self, background: bool = False, n_way: int = 5, transform=TRANSFORM):
        self.base = Omniglot(root="./omniglot_data", background=background,
                              download=True, transform=transform)
        self.class_to_indices = _index_by_class(self.base)
        # only keep classes with at least 2 samples (need support + query)
        self.classes = [c for c, idxs in self.class_to_indices.items() if len(idxs) >= 2]
        self.n_way = n_way

    def sample_episode(self):
        chosen_classes = random.sample(self.classes, self.n_way)
        correct_class_idx = random.randrange(self.n_way)

        support_imgs = []
        query_img = None
        for i, cls in enumerate(chosen_classes):
            idxs = self.class_to_indices[cls]
            if i == correct_class_idx:
                support_i, query_i = random.sample(idxs, 2)
                query_img, _ = self.base[query_i]
            else:
                support_i = random.choice(idxs)
            img, _ = self.base[support_i]
            support_imgs.append(img)

        support_batch = torch.stack(support_imgs)          # (N, 1, 105, 105)
        query_batch = query_img.unsqueeze(0).repeat(self.n_way, 1, 1, 1)  # (N, 1, 105, 105)

        return support_batch, query_batch, correct_class_idx
