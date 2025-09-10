import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
import torch

# Import custom modules from the 'src' directory
from src import config as cfg
from src import pinecone_db
from src import clustering
from src.models import load_models, to_ascii_id, l2norm
from src.pdf_processing import extract_text_from_pdf, extract_images_from_pdf
from src.text_similarity import compute_hybrid_similarity, remove_template_text
from src.image_similarity import compute_average_image_similarity, compute_phash_hex
from pythainlp.tokenize import word_tokenize

# --- Page Configuration ---
st.set_page_config(
    page_title="PDF Similarity Analysis with Pinecone",
    page_icon="https://www.kku.ac.th/wp-content/uploads/2023/04/logo-kku-color.png",
    layout="wide"
)

# --- Load Models & Connect to DB (cached for performance) ---
try:
    # **FIX:** Call the centralized model loading function
    # This will return all necessary variables: models, device, and dimensions
    labse_model, clip_model, clip_processor, device, labse_dim, clip_dim = load_models()
    st.sidebar.success(f"Models loaded on {device.upper()}!")

    # Connect to Pinecone Vector DB using the dimensions from load_models()
    text_index, image_index = pinecone_db.connect_pinecone(
        text_dim=labse_dim, image_dim=clip_dim)
    if text_index or image_index:
        st.sidebar.success("Pinecone connected!")

except Exception as e:
    st.sidebar.error(f"Initialization Error: {e}")
    st.exception(e)
    st.stop()


def main():
    st.title("🔍 PDF Similarity Analysis System (with Pinecone)")

    with st.sidebar:
        st.header("⚙️ Configuration")
        processing_mode = st.selectbox(
            "Processing Mode:",
            options=[1, 2, 3],
            format_func=lambda x: {1: "📝 Text Only",
                                   2: "🖼️ Image Only", 3: "📝🖼️ Text + Image"}[x]
        )

        use_template, template_file = False, None
        if processing_mode in [1, 3]:
            if st.checkbox("เลือกใช้ Template เพื่อลบข้อความซ้ำ หรือข้อความจากโจทย์"):
                use_template = True
                template_file = st.file_uploader(
                    "Upload Template PDF", type=['pdf'])

        text_weight = cfg.DEFAULT_TEXT_WEIGHT
        if processing_mode == 3:
            text_weight = st.slider(
                "Text Weight", 0.0, 1.0, cfg.DEFAULT_TEXT_WEIGHT, 0.05)

        group_threshold = st.slider(
            "Grouping Threshold", 0.5, 1.0, cfg.DEFAULT_GROUP_TH, 0.05)

    tab1, tab2 = st.tabs(["📤 Upload & Process", "📊 Results & Clusters"])

    with tab1:
        st.header("Upload PDF Files")
        uploaded_files = st.file_uploader(
            "Choose PDF files:", type=['pdf'], accept_multiple_files=True
        )

        if st.button("🚀 Start Processing & Comparison", type="primary", disabled=not uploaded_files or len(uploaded_files) < 2):
            process_and_compare(uploaded_files, processing_mode,
                                use_template, template_file, text_weight, group_threshold)

    with tab2:
        display_results()


def process_and_compare(uploaded_files, processing_mode, use_template, template_file, text_weight, group_threshold):
    progress_bar = st.progress(0, "Initializing...")
    status_text = st.empty()

    try:
        template_text = ""
        if use_template and template_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(template_file.getvalue())
                template_text = extract_text_from_pdf(tmp.name)
            os.unlink(tmp.name)

        processed_data, text_to_upsert, image_to_upsert = {}, {}, {}
        for i, file in enumerate(uploaded_files):
            doc_id = to_ascii_id(Path(file.name).stem)
            status_text.text(
                f"Processing {file.name} ({i+1}/{len(uploaded_files)})")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getvalue())
                tmp_path = tmp.name

            data = {"doc_id": doc_id, "filename": file.name}

            if processing_mode in [1, 3]:
                raw_text = extract_text_from_pdf(tmp_path)
                clean_text = remove_template_text(
                    raw_text, template_text) if use_template else raw_text
                tokens = word_tokenize(clean_text, engine="newmm")
                tokenized_text = " ".join(tokens)
                text_emb = l2norm(labse_model.encode(
                    tokenized_text, convert_to_numpy=True).astype(np.float32))
                data.update({"raw_text": clean_text,
                             "text_embedding": text_emb})
                text_to_upsert[doc_id] = {"embedding": text_emb}

            if processing_mode in [2, 3]:
                pil_images = extract_images_from_pdf(tmp_path)
                image_items = []
                for img in pil_images:
                    inputs = clip_processor(
                        images=img, return_tensors="pt").to(device)
                    with torch.no_grad():
                        img_emb = clip_model.get_image_features(
                            **inputs).squeeze(0)
                    img_emb_norm = l2norm(
                        img_emb.cpu().numpy().astype(np.float32))
                    image_items.append(
                        {"embedding": img_emb_norm, "phash": compute_phash_hex(img)})
                data["image_items"] = image_items
                image_to_upsert[doc_id] = {"items": image_items}

            processed_data[doc_id] = data
            os.unlink(tmp_path)
            progress_bar.progress((i + 1) / len(uploaded_files))

        pinecone_db.upsert_text_vectors(text_index, text_to_upsert)
        pinecone_db.upsert_image_vectors(image_index, image_to_upsert)

        status_text.text("Comparing document pairs...")
        results = []
        doc_ids = list(processed_data.keys())
        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                doc1 = processed_data[doc_ids[i]]
                doc2 = processed_data[doc_ids[j]]

                # --- Initialize all scores ---
                text_sim, img_sim = 0.0, 0.0
                semantic_sim, lexical_sim = 0.0, 0.0
                img_comp = {}

                if processing_mode in [1, 3]:
                    text_sim, semantic_sim, lexical_sim = compute_hybrid_similarity(
                        doc1["raw_text"], doc2["raw_text"],
                        doc1["text_embedding"], doc2["text_embedding"]
                    )

                if processing_mode in [2, 3]:
                    img_comp = compute_average_image_similarity(
                        doc1.get("image_items", []), doc2.get("image_items", []))
                    img_sim = img_comp.get("ensemble_avg", 0.0)

                if processing_mode == 1:
                    final_score = text_sim
                elif processing_mode == 2:
                    final_score = img_sim
                else:
                    final_score = (text_weight * text_sim) + \
                        ((1 - text_weight) * img_sim)

                level = "เหมือนมาก" if final_score >= 0.9 else "คล้ายสูง" if final_score >= 0.75 else "คล้ายระดับกลาง" if final_score >= 0.6 else "ไม่คล้าย"

                # --- Store all detailed scores ---
                result_pair = {
                    "doc_1": doc1["doc_id"],
                    "doc_2": doc2["doc_id"],
                    "final_score": final_score,
                    "level": level,
                    "text_hybrid_similarity": text_sim,
                    "text_semantic_similarity": semantic_sim,
                    "text_lexical_similarity": lexical_sim,
                    "image_similarity": img_sim
                }
                # Add detailed image scores from the 'img_comp' dictionary
                result_pair.update(img_comp)
                results.append(result_pair)

        st.session_state.results_df = pd.DataFrame(
            sorted(results, key=lambda x: x['final_score'], reverse=True))

        status_text.text("Generating clusters...")
        st.session_state.clusters = clustering.cluster_by_threshold(
            st.session_state.results_df, group_threshold)
        st.session_state.processed_filenames = {
            d["doc_id"]: d["filename"] for d in processed_data.values()}

        status_text.text("✅ Done!")
        st.success("Processing, comparison, and clustering complete!")
        st.balloons()
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        st.exception(e)


def display_results():
    if 'results_df' not in st.session_state or st.session_state.results_df.empty:
        st.info("Process files to see results and clusters here.")
        return

    df = st.session_state.results_df
    clusters = st.session_state.clusters
    filenames = st.session_state.processed_filenames

    st.subheader("📊 Detailed Comparison Results")

    # --- NEW: Detailed per-pair display ---
    for index, row in df.iterrows():
        expander_title = (
            f"**{filenames.get(row['doc_1'])}** vs **{filenames.get(row['doc_2'])}** | "
            f"Final Score: **{row['final_score']:.2%}** ({row['level']})"
        )
        with st.expander(expander_title):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📝 Text Analysis")
                # This check is more robust: it checks if a text-specific score exists and has a value.
                if "text_semantic_similarity" in row and pd.notna(row["text_semantic_similarity"]):
                    st.metric("Hybrid Score",
                              f"{row['text_hybrid_similarity']:.4f}")
                    st.write(
                        f"- Semantic (Embedding): `{row['text_semantic_similarity']:.4f}`")
                    st.write(
                        f"- Lexical (TF-IDF): `{row['text_lexical_similarity']:.4f}`")
                else:
                    st.info("No text comparison in this mode.")

            with col2:
                st.markdown("#### 🖼️ Image Analysis")
                # --- FIX: Check for 'total_pairs' existence ---
                # This key is only added if image comparison was performed, making it a reliable check
                # regardless of the similarity score (even if it's 0).
                if "total_pairs" in row:
                    st.metric("Ensemble Score (Avg)",
                              f"{row['image_similarity']:.4f}")
                    st.write(
                        f"- Embedding Cosine (Avg): `{row.get('cosine_avg', 0.0):.4f}`")
                    st.write(
                        f"- pHash Similarity (Avg): `{row.get('phash_avg', 0.0):.4f}`")
                    st.write(
                        f"- Matched Pairs: `{row.get('matched_images', 0)} / {row.get('total_pairs', 0)}`")
                else:
                    st.info("No image comparison in this mode.")

    # --- Clustering display remains the same ---
    st.subheader("🔗 Document Clusters")
    if not clusters:
        st.warning("No clusters found with the current threshold.")
    else:
        for i, cluster in enumerate(clusters):
            with st.expander(f"**Cluster {i+1}** ({len(cluster)} documents)"):
                for doc_id in cluster:
                    st.markdown(f"- `{filenames.get(doc_id, doc_id)}`")


if __name__ == "__main__":
    main()
