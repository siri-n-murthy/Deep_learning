# One-Shot Handwritten Character Recognition using a Siamese Network

This project trains a Siamese neural network to compare handwritten characters and recognize them from very few examples. Instead of learning a fixed output layer for a closed set of characters, the model learns a similarity function: it measures how close two images are in an embedding space.

## Why this is useful

Most traditional classifiers need many labeled examples per class. Omniglot is a few-shot learning dataset: each character class has only a handful of samples. A standard classifier would need retraining for every new class, but a Siamese network can compare a new image against reference examples and decide which class it is most similar to.

## Project idea

1. Data: Omniglot contains handwritten characters from many alphabets.
2. Model: Two images pass through the same CNN (shared weights) to produce embeddings.
3. Training: If two images are the same character, their embeddings are pulled close together. If they are different, they are pushed apart using contrastive loss.
4. Testing: The model sees novel characters it has never encountered before and compares them to a small reference set.
5. Usage: A user uploads one image or two images, and the app predicts either the closest match or whether the two images are the same.

## Project structure

```text
app.py          - Streamlit app for same/different comparison and single-character recognition
compare.py      - command-line comparison script for two images
data.py         - Omniglot dataset loading and pair generation
evaluate.py     - N-way one-shot evaluation on unseen classes
find_threshold.py - calibrates the distance threshold for same/different decisions
loss.py         - contrastive loss implementation
model.py        - shared CNN + Siamese wrapper
recognize.py    - nearest-neighbor single-character recognition helper and CLI
train.py        - training loop
requirements.txt - project dependencies
```

## How it works

A Siamese network learns a distance metric rather than a fixed label map. For each image, it produces an embedding vector. The model is trained so that:

- same character -> small distance
- different character -> large distance

At test time, a new image is embedded and compared to reference embeddings using Euclidean distance. The nearest match is treated as the predicted class.

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py --epochs 20 --batch-size 64
```

This downloads Omniglot automatically if needed and saves the trained weights to `siamese_model.pt`.

## Evaluation

```bash
python evaluate.py --n-way 5 --num-episodes 200
python evaluate.py --n-way 20 --num-episodes 200
```

These commands evaluate one-shot recognition on unseen classes.

## Single-character recognition

### Streamlit app

```bash
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open the browser at http://127.0.0.1:8501 and use the app to:

- compare two images
- recognize a single character against a reference gallery

### Command line

```bash
python recognize.py --image my_character.png --model-path siamese_model.pt --max-classes 20
```

This prints the predicted class and nearest distance.

## Why this project matters

This project demonstrates a classic few-shot learning idea: learning a comparison function instead of memorizing a fixed label space. That makes it relevant to fingerprint matching, face verification, signature recognition, and other real-world tasks where new classes appear with limited data.

## Future improvements

- use cosine distance instead of Euclidean distance
- add data augmentation for better generalization
- try triplet loss instead of contrastive loss
- build a true alphabet-specific classifier using a known letter set
- visualize embedding clusters with t-SNE or PCA

## One-line pitch

I trained a Siamese neural network to compare handwritten characters so it can recognize new ones from just a few examples.
