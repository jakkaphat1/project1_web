from __future__ import annotations
from typing import Dict
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from .config import PINECONE_API_KEY, PINECONE_INDEX_TEXT, PINECONE_INDEX_IMAGE

try:
    import streamlit as st
except ImportError:
    st = None


def connect_pinecone(text_dim: int, image_dim: int):
    """Connects to Pinecone and ensures indexes exist."""
    if not PINECONE_API_KEY:
        if st:
            st.warning(
                "PINECONE_API_KEY not found. Skipping Pinecone connection.")
        return None, None

    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = {i.name for i in pc.list_indexes()}

    if PINECONE_INDEX_TEXT not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX_TEXT, dimension=text_dim, metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    if PINECONE_INDEX_IMAGE not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX_IMAGE, dimension=image_dim, metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    return pc.Index(PINECONE_INDEX_TEXT), pc.Index(PINECONE_INDEX_IMAGE)


def upsert_text_vectors(index, text_results: Dict):
    """Upserts text vectors into the specified Pinecone index."""
    if not index or not text_results:
        return

    vectors_to_upsert = [
        {"id": f"text::{doc_id}", "values": data["embedding"].tolist()}
        for doc_id, data in text_results.items()
    ]
    if vectors_to_upsert:
        index.upsert(vectors=vectors_to_upsert)
        if st:
            st.toast(
                f"Upserted {len(vectors_to_upsert)} text vectors to Pinecone.")


def upsert_image_vectors(index, image_results: Dict):
    """Upserts image vectors into the specified Pinecone index."""
    if not index or not image_results:
        return

    vectors_to_upsert = []
    for doc_id, data in image_results.items():
        for i, item in enumerate(data.get("items", [])):
            vectors_to_upsert.append({
                "id": f"image::{doc_id}::{i}",
                "values": item["embedding"].tolist(),
                "metadata": {"phash": item["phash"]}
            })

    if vectors_to_upsert:
        index.upsert(vectors=vectors_to_upsert)
        if st:
            st.toast(
                f"Upserted {len(vectors_to_upsert)} image vectors to Pinecone.")
