# # from __future__ import annotations
# # from functools import lru_cache as cache_resource
# # from typing import Tuple
# # import torch
# # from sentence_transformers import SentenceTransformer
# # from transformers import CLIPModel, CLIPProcessor
# # from PIL import Image
# # import streamlit as st

# # try:
# #     import streamlit as st
# #     cache_resource = st.cache_resource
# # except Exception:
# #     pass


# # @st.cache_resource(show_spinner=False)
# # def load_models() -> Tuple[SentenceTransformer, CLIPModel, CLIPProcessor, str, int, int]:
# #     labse = SentenceTransformer("sentence-transformers/LaBSE")
# #     clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# #     clip_processor = CLIPProcessor.from_pretrained(
# #         "openai/clip-vit-base-patch32")
# #     device = "cuda" if torch.cuda.is_available() else "cpu"
# #     clip_model = clip_model.to(device)


# # # Dimensions
# #     text_dim = getattr(
# #         labse, "get_sentence_embedding_dimension", lambda: 768)()
# #     with torch.no_grad():
# #         dummy = Image.new("RGB", (224, 224), color=(128, 128, 128))
# #         tmp = clip_processor(images=dummy, return_tensors="pt").to(device)
# #         emb = clip_model.get_image_features(
# #             **tmp).squeeze(0).detach().cpu().numpy()
# #         image_dim = emb.shape[0]  # usually 512 for ViT-B/32
# #     return labse, clip_model, clip_processor, device, text_dim, image_dim

# import streamlit as st
# import numpy as np
# from slugify import slugify
# import torch
# from sentence_transformers import SentenceTransformer
# from transformers import CLIPProcessor, CLIPModel


# def l2norm(x: np.ndarray) -> np.ndarray:
#     """Computes L2 normalization."""
#     n = np.linalg.norm(x)
#     # Add a small epsilon to avoid division by zero
#     return x / (n + 1e-12)


# def to_ascii_id(filename: str) -> str:
#     """Generates a clean, ASCII-safe ID from a filename."""
#     return slugify(filename, separator="_")


# @st.cache_resource
# def load_models():
#     """
#     Loads all required machine learning models and caches them.
#     This function will download the models on its first run.
#     """
#     st.info("Loading models... This might take a moment on the first run.")

#     # Determine the device to run the models on (GPU if available, otherwise CPU)
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # Load the LaBSE model for text embeddings
#     # Model from: https://huggingface.co/sentence-transformers/LaBSE
#     labse_model = SentenceTransformer(
#         "sentence-transformers/LaBSE", device=device)

#     # Load the CLIP model for image embeddings
#     # Model from: https://huggingface.co/openai/clip-vit-large-patch14
#     clip_model = CLIPModel.from_pretrained(
#         "openai/clip-vit-large-patch14").to(device)
#     clip_processor = CLIPProcessor.from_pretrained(
#         "openai/clip-vit-large-patch14")

#     return labse_model, clip_model, clip_processor, device


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
