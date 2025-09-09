# from __future__ import annotations
# import io
# import fitz  # PyMuPDF
# from pdf2image import convert_from_path
# import pytesseract
# import numpy as np
# from PIL import Image
# import imagehash
# from typing import List
# from .utils import l2norm
# from .config import MIN_IMG_AREA_RATIO, MIN_IMG_PIXELS, USE_PAGE_RENDER_FALLBACK
# from transformers import CLIPModel, CLIPProcessor
# from sentence_transformers import SentenceTransformer
# import torch

# try:
#     import streamlit as st
# except Exception:  # pragma: no cover
#     st = None  # type: ignore


# def ocr_pages_to_text(images, lang="tha+eng") -> str:
#     texts = [pytesseract.image_to_string(img, lang=lang) for img in images]
#     return "\n".join(texts).strip()


# class ImageItem:
#     def __init__(self, embedding: np.ndarray, phash_hex: str):
#         self.embedding = embedding
#         self.phash_hex = phash_hex


# class TextResult:
#     def __init__(self, doc_id: str, raw_text: str, clean_text: str, embedding: np.ndarray):
#         self.doc_id = doc_id
#         self.raw_text = raw_text
#         self.clean_text = clean_text
#         self.embedding = embedding


# class ImageResult:
#     def __init__(self, doc_id: str, items: List[ImageItem]):
#         self.doc_id = doc_id
#         self.items = items


# def extract_text_and_embed(pdf_bytes: bytes, labse_model: SentenceTransformer) -> tuple[str, np.ndarray]:
#     doc = fitz.open(stream=pdf_bytes, filetype="pdf")
#     text = "\n".join([page.get_text("text") for page in doc]).strip()

#     if not text:
#         try:
#             # pdf2image normally takes file path; for BytesIO environments this may fail.
#             # If it fails, we just warn and return empty text.
#             images = convert_from_path(io.BytesIO(
#                 pdf_bytes))  # may fail in some envs
#             text = ocr_pages_to_text(images, lang="tha+eng")
#         except Exception as e:
#             if st:
#                 st.warning(f"[OCR] ล้มเหลว: {e}")
#             text = ""

#     text_embedding = labse_model.encode(text)
#     text_embedding = np.asarray(text_embedding, dtype=np.float32)
#     text_embedding = l2norm(text_embedding)
#     return text, text_embedding


# def compute_phash_hex(pil_img: Image.Image) -> str:
#     return str(imagehash.phash(pil_img))


# def phash_hamming_similarity(hex1: str, hex2: str) -> float:
#     if not hex1 or not hex2:
#         return 0.0
#     h1 = imagehash.hex_to_hash(hex1)
#     h2 = imagehash.hex_to_hash(hex2)
#     d = h1 - h2
#     nbits = h1.hash.size
#     return 1.0 - (d / nbits)


# def extract_images_and_embed(pdf_bytes: bytes, clip_model: CLIPModel, clip_processor: CLIPProcessor, device: str) -> list[ImageItem]:
#     doc = fitz.open(stream=pdf_bytes, filetype="pdf")
#     images_to_embed: list[Image.Image] = []

#     for page_num in range(len(doc)):
#         page = doc.load_page(page_num)
#         page_area = (page.rect.width * page.rect.height) if page.rect else 0
#         found_embedded = False

#         # 1) embedded images
#         if page_area > 0:
#             for img_info in page.get_images(full=True):
#                 xref = img_info[0]
#                 try:
#                     bbox = page.get_image_bbox(img_info)
#                     img_area = bbox.width * bbox.height
#                 except Exception:
#                     img_area = 0

#                 take_this = False
#                 if page_area > 0 and img_area > 0 and (img_area / page_area) >= MIN_IMG_AREA_RATIO:
#                     take_this = True
#                 else:
#                     base_image = doc.extract_image(xref)
#                     w = base_image.get("width", 0)
#                     h = base_image.get("height", 0)
#                     if (w * h) >= MIN_IMG_PIXELS:
#                         take_this = True

#                 if take_this:
#                     base_image = doc.extract_image(xref)
#                     img_bytes = base_image["image"]
#                     img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
#                     images_to_embed.append(img)
#                     found_embedded = True

#         # 2) drawings fallback (crop)
#         if not found_embedded:
#             total_drawings_bbox = fitz.Rect()
#             drawings = page.get_drawings()
#             for path in drawings:
#                 total_drawings_bbox.include_rect(path['rect'])
#             if not total_drawings_bbox.is_empty and (total_drawings_bbox.width > 20 and total_drawings_bbox.height > 20):
#                 pix = page.get_pixmap(dpi=150, clip=total_drawings_bbox)
#                 img = Image.frombytes(
#                     "RGB", [pix.width, pix.height], pix.samples)
#                 images_to_embed.append(img)
#                 found_embedded = True

#         # 3) full page render if enabled
#         if not found_embedded and USE_PAGE_RENDER_FALLBACK:
#             pix = page.get_pixmap(dpi=150)
#             img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#             images_to_embed.append(img)

#     # Embed & pHash
#     items: list[ImageItem] = []
#     for img in images_to_embed:
#         inputs = clip_processor(images=img.convert(
#             "RGB"), return_tensors="pt").to(device)
#         with torch.no_grad():
#             img_embedding = clip_model.get_image_features(**inputs).squeeze(0)
#         emb_np = img_embedding.detach().cpu().numpy().astype(np.float32)
#         emb_np = l2norm(emb_np)
#         phash_hex = compute_phash_hex(img)
#         items.append(ImageItem(embedding=emb_np, phash_hex=phash_hex))

#     return items


from __future__ import annotations
import fitz  # PyMuPDF
from PIL import Image
import io
from typing import List
from pathlib import Path

# Import config values
from .config import MIN_IMG_AREA_RATIO, MIN_IMG_PIXELS, USE_PAGE_RENDER_FALLBACK


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts all text from a given PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""


def extract_images_from_pdf(pdf_path: str) -> List[Image.Image]:
    """
    Extracts meaningful images from a PDF using a sophisticated, page-by-page logic.

    Extraction Strategy (per page):
    1.  **Embedded Images:** Look for standard embedded images (PNG, JPEG) that are
        large enough based on pixel dimensions or page area ratio.
    2.  **Vector Drawings:** If no embedded images are found on the page, look for
        vector graphics (drawings, charts) and render them as an image.
    3.  **Full Page Fallback:** If still nothing is found and the option is enabled
        in config, render the entire page as a last resort.
    """
    images_to_embed = []
    doc = fitz.open(pdf_path)

    for page in doc:
        found_on_page = False
        page_area = page.rect.width * page.rect.height if page.rect else 0

        # --- Step 1: Try to find significant embedded images on this page ---
        if page_area > 0:
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                base_image = doc.extract_image(xref)

                # Check if the image is large enough
                w, h = base_image.get("width", 0), base_image.get("height", 0)
                try:
                    bbox = page.get_image_bbox(img_info)
                    img_area = bbox.width * bbox.height
                except Exception:
                    img_area = 0

                take_this = False
                if (img_area / page_area) >= MIN_IMG_AREA_RATIO:
                    take_this = True
                elif (w * h) >= MIN_IMG_PIXELS:
                    take_this = True

                if take_this:
                    try:
                        img_bytes = base_image["image"]
                        pil_image = Image.open(
                            io.BytesIO(img_bytes)).convert("RGB")
                        images_to_embed.append(pil_image)
                        found_on_page = True
                    except Exception as e:
                        print(
                            f"Warning: Could not process embedded image xref {xref}: {e}")

        # --- Step 2: If no embedded images, look for vector drawings ---
        if not found_on_page:
            drawings = page.get_drawings()
            if drawings:
                total_drawings_bbox = fitz.Rect()
                for path in drawings:
                    total_drawings_bbox.include_rect(path['rect'])

                # Render drawings if they form a significant area
                if not total_drawings_bbox.is_empty and (total_drawings_bbox.width > 20 and total_drawings_bbox.height > 20):
                    pix = page.get_pixmap(dpi=150, clip=total_drawings_bbox)
                    pil_image = Image.frombytes(
                        "RGB", [pix.width, pix.height], pix.samples)
                    images_to_embed.append(pil_image)
                    found_on_page = True

        # --- Step 3: Fallback to rendering the whole page if needed ---
        if not found_on_page and USE_PAGE_RENDER_FALLBACK:
            print(
                f"No content found on page {page.number} of {Path(pdf_path).name}. Rendering full page.")
            pix = page.get_pixmap(dpi=150)
            pil_image = Image.frombytes(
                "RGB", [pix.width, pix.height], pix.samples)
            images_to_embed.append(pil_image)

    return images_to_embed
