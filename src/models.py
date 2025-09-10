import streamlit as st
import numpy as np
from slugify import slugify
import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import os
from pathlib import Path  # Import pathlib for robust path handling


def l2norm(x: np.ndarray) -> np.ndarray:
    """Computes L2 normalization."""
    n = np.linalg.norm(x)
    return x / (n + 1e-12)


def to_ascii_id(filename: str) -> str:
    """Generates a clean, ASCII-safe ID from a filename."""
    return slugify(filename, separator="_")


@st.cache_resource
def load_models():

    project_root = Path(__file__).parent.parent
    cache_dir = project_root / ".model_cache"
    os.makedirs(cache_dir, exist_ok=True)

    # Set environment variables for both libraries to use this new cache folder
    os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(cache_dir)
    os.environ['HF_HOME'] = str(cache_dir)
    # -------------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load LaBSE model and get its dimension
    # The cache_folder argument is still good practice to be explicit
    labse_model = SentenceTransformer(
        "sentence-transformers/LaBSE", device=device, cache_folder=str(cache_dir))
    labse_dim = labse_model.get_sentence_embedding_dimension()

    # Load CLIP model and get its dimension
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14", cache_dir=str(cache_dir)).to(device)
    clip_processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-large-patch14", cache_dir=str(cache_dir))
    clip_dim = clip_model.config.projection_dim

    return labse_model, clip_model, clip_processor, device, labse_dim, clip_dim
