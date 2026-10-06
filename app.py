"""
Streamlit web app: upload two images, see them side by side, get the
embedding distance and a same/different verdict from the trained
Siamese network.

Usage:
    streamlit run app.py

This opens a local web page (usually http://localhost:8501) in your browser.
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image

from model import SiameseNetwork
from data import TRANSFORM
from recognize import predict_single_character


st.set_page_config(page_title="Siamese Character Verifier", page_icon="🔤", layout="centered")


@st.cache_resource
def load_model(model_path: str, device: str):
    model = SiameseNetwork(embedding_dim=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def preprocess(pil_image: Image.Image):
    gray = pil_image.convert("L")
    tensor = TRANSFORM(gray).unsqueeze(0)  # (1, 1, 105, 105)
    return tensor


def get_distance(model, tensor_a, tensor_b, device):
    with torch.no_grad():
        emb_a = model.embed(tensor_a.to(device))
        emb_b = model.embed(tensor_b.to(device))
        distance = torch.nn.functional.pairwise_distance(emb_a, emb_b, p=2).item()
    return distance


def validate_character_image(img: Image.Image) -> bool:
    """Allow handwritten script and isolated character images while rejecting
    natural photos or other non-character content.

    The project targets hand-written text and symbols, so we accept images with
    strong stroke-like structure, high contrast, and low overall colorfulness.
    """
    try:
        if img.mode in ("RGBA", "RGB"):
            rgb = np.asarray(img.convert("RGB"))
            channel_span = np.max(rgb, axis=2) - np.min(rgb, axis=2)
            colorfulness = float(np.mean(channel_span))
            if colorfulness > 55:
                return False
        gray = img.convert("L")
    except Exception:
        return False

    arr = np.asarray(gray)
    if arr.ndim != 2:
        return False
    if arr.shape[0] < 30 or arr.shape[1] < 30:
        return False

    flat = arr.astype(np.float32).ravel()
    contrast = float(np.std(flat))
    dark_fraction = float(np.mean(flat < 180))
    light_fraction = float(np.mean(flat > 200))

    if contrast < 18:
        return False
    if dark_fraction + light_fraction < 0.12:
        return False

    gy, gx = np.gradient(arr.astype(np.float32))
    edge_strength = np.abs(gx) + np.abs(gy)
    edge_density = float(np.mean(edge_strength > 10))

    if edge_density < 0.02:
        return False

    # Many natural photos have smooth, continuous gradients and stronger full-scene
    # color variation than handwriting. Papers and handwritten script usually have
    # stronger structure and more concentrated ink strokes.
    if edge_density > 0.60:
        return False

    return True


# ---- UI ----

st.title("🔤 Siamese Network Character Verifier")
st.caption("Only upload handwritten character images in the Omniglot style. Natural photos or general scenes are rejected.")

device = "cuda" if torch.cuda.is_available() else "cpu"

with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Distance threshold",
        min_value=0.0, max_value=3.0, value=0.6, step=0.01,
        help="Distance below this = predicted SAME. Run find_threshold.py to calibrate "
             "this properly instead of guessing."
    )
    reference_classes = st.slider(
        "Reference classes",
        min_value=5,
        max_value=100,
        value=20,
        step=5,
        help="How many Omniglot character classes to compare against when recognizing one image."
    )
    model_path = st.text_input("Model file", value="siamese_model.pt")

col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("Image A", type=["png", "jpg", "jpeg"], key="a")
    if file_a:
        img_a = Image.open(file_a)
        if not validate_character_image(img_a):
            st.warning("Please upload a handwritten character image, not a general photo.")
        else:
            st.image(img_a, caption="Image A", use_container_width=True)

with col2:
    file_b = st.file_uploader("Image B", type=["png", "jpg", "jpeg"], key="b")
    if file_b:
        img_b = Image.open(file_b)
        if not validate_character_image(img_b):
            st.warning("Please upload a handwritten character image, not a general photo.")
        else:
            st.image(img_b, caption="Image B", use_container_width=True)

st.divider()

if file_a and file_b:
    if st.button("Compare", type="primary", use_container_width=True):
        if not validate_character_image(img_a) or not validate_character_image(img_b):
            st.error("Only handwritten character images are allowed. Please upload Omniglot-style letters or symbols, not natural photos.")
            st.stop()

        try:
            model = load_model(model_path, device)
        except FileNotFoundError:
            st.error(f"Couldn't find '{model_path}'. Make sure you've run train.py first "
                      "and the .pt file is in this folder.")
            st.stop()

        tensor_a = preprocess(img_a)
        tensor_b = preprocess(img_b)
        distance = get_distance(model, tensor_a, tensor_b, device)

        is_same = distance < threshold

        st.subheader("Result")
        m1, m2 = st.columns(2)
        m1.metric("Embedding distance", f"{distance:.4f}")
        m2.metric("Threshold", f"{threshold:.2f}")

        if is_same:
            st.success("✅ Predicted: **SAME character**")
        else:
            st.warning("❌ Predicted: **DIFFERENT characters**")

        # simple visual bar showing distance relative to threshold
        st.write("Distance vs. threshold:")
        bar_max = max(distance, threshold) * 1.3
        st.progress(min(distance / bar_max, 1.0))

st.divider()

st.subheader("🔠 Recognize one character at a time")
st.caption("Only Omniglot-style handwritten symbols are accepted. Photos and other images are rejected.")

single_file = st.file_uploader("Single character image", type=["png", "jpg", "jpeg"], key="single")

if single_file:
    single_img = Image.open(single_file)
    if not validate_character_image(single_img):
        st.warning("Please upload a handwritten character image, not a general photo.")
    else:
        st.image(single_img, caption="Input image", use_container_width=True)

    if st.button("Recognize", type="primary", use_container_width=True):
        if not validate_character_image(single_img):
            st.error("Only handwritten character images are allowed. Please upload a single letter or symbol, not a natural photo.")
            st.stop()

        try:
            model = load_model(model_path, device)
        except FileNotFoundError:
            st.error(f"Couldn't find '{model_path}'. Make sure you've run train.py first "
                      "and the .pt file is in this folder.")
            st.stop()

        predicted_label, predicted_distance, _, _ = predict_single_character(
            model,
            single_img,
            device=device,
            max_classes=reference_classes,
        )

        st.subheader("Recognition result")
        col1, col2 = st.columns(2)
        col1.metric("Predicted class", predicted_label)
        col2.metric("Nearest distance", f"{predicted_distance:.4f}")
        st.success(f"✅ Closest match: class {predicted_label}")
else:
    st.info("Upload one image to recognize a single character.")
