"""
Evaluates the trained Siamese network on N-way one-shot classification,
using the Omniglot evaluation split (classes never seen during training).

This is the metric that actually matters for this kind of model -- plain
pair accuracy can look deceptively good/bad, but N-way one-shot accuracy
directly measures "can this model recognize a brand-new character class
from a single example?", which is the whole point of a verification network.

Usage:
    python evaluate.py --n-way 5 --num-episodes 200
    python evaluate.py --n-way 20 --num-episodes 200
"""

import argparse

import torch

from data import NWayOneShotTask
from model import SiameseNetwork


def run_evaluation(model, task_generator: NWayOneShotTask, num_episodes: int, device: str):
    model.eval()
    correct = 0

    with torch.no_grad():
        for _ in range(num_episodes):
            support_batch, query_batch, correct_idx = task_generator.sample_episode()
            support_batch, query_batch = support_batch.to(device), query_batch.to(device)

            support_emb = model.embed(support_batch)   # (N, D)
            query_emb = model.embed(query_batch)        # (N, D) -- same query repeated N times

            distances = torch.nn.functional.pairwise_distance(support_emb, query_emb, p=2)
            predicted_idx = torch.argmin(distances).item()

            if predicted_idx == correct_idx:
                correct += 1

    accuracy = correct / num_episodes
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-way", type=int, default=5,
                         help="Number of classes per episode (5-way, 20-way, etc.)")
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--model-path", type=str, default="siamese_model.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SiameseNetwork(embedding_dim=64).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    task_generator = NWayOneShotTask(background=False, n_way=args.n_way)

    accuracy = run_evaluation(model, task_generator, args.num_episodes, device)
    print(f"{args.n_way}-way one-shot accuracy over {args.num_episodes} episodes: "
          f"{accuracy * 100:.2f}%")
