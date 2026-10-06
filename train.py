import argparse

import torch
from torch.utils.data import DataLoader

from data import OmniglotPairDataset
from model import SiameseNetwork
from loss import ContrastiveLoss


def train(epochs: int, batch_size: int, lr: float, margin: float,
          pairs_per_epoch: int, device: str):

    print("Loading data (downloads Omniglot on first run)...")
    train_dataset = OmniglotPairDataset(background=True, num_pairs=pairs_per_epoch)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                               shuffle=True, num_workers=2)

    model = SiameseNetwork(embedding_dim=64).to(device)
    criterion = ContrastiveLoss(margin=margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (img_a, img_b, label) in enumerate(train_loader):
            img_a, img_b, label = img_a.to(device), img_b.to(device), label.to(device)

            optimizer.zero_grad()
            emb_a, emb_b = model(img_a, img_b)
            loss = criterion(emb_a, emb_b, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch}/{epochs}  Batch {batch_idx}/{len(train_loader)}  "
                      f"Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        print(f"=== Epoch {epoch} done. Avg loss: {avg_loss:.4f} ===")

    torch.save(model.state_dict(), "siamese_model.pt")
    print("Model saved to siamese_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--pairs-per-epoch", type=int, default=20000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
          margin=args.margin, pairs_per_epoch=args.pairs_per_epoch, device=device)
