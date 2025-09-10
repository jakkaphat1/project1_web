from __future__ import annotations
import os
import streamlit as st
# from dotenv import load_dotenv
# load_dotenv()


# try:
#     import streamlit as st
#     _S = getattr(st, "secrets", {})
# except Exception:
#     _S = {}

# PINECONE_API_KEY: str | None = _S.get(
#     "PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY")
# PINECONE_INDEX_TEXT: str = _S.get("PINECONE_INDEX_TEXT") or os.getenv(
#     "PINECONE_INDEX_TEXT", "pdf-text")
# PINECONE_INDEX_IMAGE: str = _S.get("PINECONE_INDEX_IMAGE") or os.getenv(
#     "PINECONE_INDEX_IMAGE", "pdf-image")

def _get(name, default=None):
    return st.secrets.get(name, os.getenv(name, default))


PINECONE_API_KEY = _get("PINECONE_API_KEY")
PINECONE_INDEX_TEXT = _get("PINECONE_INDEX_TEXT", "pdf-text")
PINECONE_INDEX_IMAGE = _get("PINECONE_INDEX_IMAGE", "pdf-image")

# Heuristics for image extraction
MIN_IMG_AREA_RATIO: float = 0.05
MIN_IMG_PIXELS: int = 64 * 64
USE_PAGE_RENDER_FALLBACK: bool = False


# UI defaults
DEFAULT_ALPHA = 0.5  # text semantic weight
DEFAULT_W_EMB = 0.7  # image CLIP weight
DEFAULT_PHASH_PREFILTER = 0.35
DEFAULT_IMG_MATCH_TH = 0.80
DEFAULT_TEXT_WEIGHT = 0.5
DEFAULT_GROUP_TH = 0.80
