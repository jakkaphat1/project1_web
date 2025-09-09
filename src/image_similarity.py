# from __future__ import annotations
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from .pdf_processing import ImageItem, phash_hamming_similarity
# import numpy as _np


# def pair_image_similarity_with_phash(emb1: np.ndarray, emb2: np.ndarray, phash_hex1: str, phash_hex2: str, w_emb: float = 0.7, w_ph: float = 0.3) -> float:
#     cos = float(cosine_similarity([emb1], [emb2])[0][0])
#     phs = float(phash_hamming_similarity(phash_hex1, phash_hex2))
#     return (w_emb * cos) + (w_ph * phs)


# def compute_average_image_similarity(imgs1: list[ImageItem], imgs2: list[ImageItem], w_emb=0.7, w_ph=0.3, phash_prefilter=0.35, match_threshold=0.80):
#     if not imgs1 or not imgs2:
#         return {"cosine_avg": 0.0, "phash_avg": 0.0, "ensemble_avg": 0.0, "matched_images": 0, "total_pairs": 0}

#     cos_list, ph_list, ens_list = [], [], []
#     matched = 0
#     total_pairs = 0

#     for a in imgs1:
#         for b in imgs2:
#             phsim = phash_hamming_similarity(a.phash_hex, b.phash_hex) if (
#                 a.phash_hex and b.phash_hex) else 0.0
#             if phsim < phash_prefilter:
#                 continue
#             cos = float(cosine_similarity([a.embedding], [b.embedding])[0][0])
#             ens = (w_emb * cos) + (w_ph * phsim)
#             cos_list.append(cos)
#             ph_list.append(phsim)
#             ens_list.append(ens)
#             total_pairs += 1
#             if ens >= match_threshold:
#                 matched += 1
#     if total_pairs == 0:
#         return {"cosine_avg": 0.0, "phash_avg": 0.0, "ensemble_avg": 0.0, "matched_images": 0, "total_pairs": 0}

#     return {
#         "cosine_avg": float(_np.mean(cos_list)),
#         "phash_avg": float(_np.mean(ph_list)),
#         "ensemble_avg": float(_np.mean(ens_list)),
#         "matched_images": int(matched),
#         "total_pairs": int(total_pairs),
#     }


import numpy as np
from PIL import Image
import imagehash
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict


def compute_phash_hex(pil_img: Image.Image) -> str:
    """Computes pHash and returns it as a hex string."""
    return str(imagehash.phash(pil_img))


def phash_hamming_similarity(hex1: str, hex2: str) -> float:
    """Calculates similarity based on pHash Hamming distance."""
    if not hex1 or not hex2:
        return 0.0
    h1 = imagehash.hex_to_hash(hex1)
    h2 = imagehash.hex_to_hash(hex2)
    d = h1 - h2
    nbits = h1.hash.size
    return 1.0 - (d / nbits)


def compute_average_image_similarity(
    image_items1: List[Dict],
    image_items2: List[Dict],
    w_emb: float = 0.7,
    w_ph: float = 0.3,
    match_threshold: float = 0.80
) -> Dict:
    """
    Calculates the average similarity between two sets of image items.
    Each item is a dict with 'embedding' and 'phash'.
    """
    if not image_items1 or not image_items2:
        return {
            "ensemble_avg": 0.0, "cosine_avg": 0.0, "phash_avg": 0.0,
            "matched_images": 0, "total_pairs": 0
        }

    cos_list, ph_list, ens_list = [], [], []
    matched_count = 0
    total_pairs = len(image_items1) * len(image_items2)

    for item1 in image_items1:
        for item2 in image_items2:
            emb1, ph1 = item1["embedding"], item1["phash"]
            emb2, ph2 = item2["embedding"], item2["phash"]

            cos_sim = cosine_similarity([emb1], [emb2])[0][0]
            phash_sim = phash_hamming_similarity(ph1, ph2)

            ensemble_score = (w_emb * cos_sim) + (w_ph * phash_sim)

            cos_list.append(cos_sim)
            ph_list.append(phash_sim)
            ens_list.append(ensemble_score)

            if ensemble_score >= match_threshold:
                matched_count += 1

    return {
        "ensemble_avg": float(np.mean(ens_list)) if ens_list else 0.0,
        "cosine_avg": float(np.mean(cos_list)) if cos_list else 0.0,
        "phash_avg": float(np.mean(ph_list)) if ph_list else 0.0,
        "matched_images": matched_count,
        "total_pairs": total_pairs
    }
