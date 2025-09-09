# from __future__ import annotations

# import numpy as np
# from difflib import SequenceMatcher
# import regex as re
# from pythainlp.tokenize import word_tokenize
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity


# def remove_template_text(student_text: str, template_text: str, threshold: float = 0.9, return_removed: bool = False):
#     student_lines = student_text.strip().split('\n')
#     template_lines = template_text.strip().split('\n')

#     result, removed = [], []
#     for s_line in student_lines:
#         match_found = any(SequenceMatcher(None, s_line.strip(
#         ), t_line.strip()).ratio() > threshold for t_line in template_lines)
#         if not match_found:
#             result.append(s_line)
#         else:
#             removed.append(s_line)

#     if return_removed:
#         return "\n".join(result).strip(), "\n".join(removed).strip()
#     return "\n".join(result).strip()


# def preprocess_text_keep_thai(text: str) -> str:
#     text = re.sub(r"[^\p{L}\p{N}\s]", " ", text)
#     text = re.sub(r"\s+", " ", text).strip()
#     return text


# def tokenize_thai(text: str) -> str:
#     # ตัดคำไทย (รวมอังกฤษ/ตัวเลข) แล้ว join ด้วย space เพื่อ feed เข้า TF-IDF
#     tokens = word_tokenize(text, engine="newmm")
#     return " ".join(tokens)


# def compute_lexical_similarity(text1: str, text2: str, ngram_range=(1, 3)) -> float:
#     t1 = tokenize_thai(preprocess_text_keep_thai(text1))
#     t2 = tokenize_thai(preprocess_text_keep_thai(text2))
#     # หมายเหตุ: เรา “pre-tokenize” แล้ว จึงให้ vectorizer treat เป็น word-level ปกติ
#     vec = TfidfVectorizer(
#         analyzer="word", ngram_range=ngram_range, token_pattern=r"(?u)\b\w+\b")
#     tfidf = vec.fit_transform([t1, t2])
#     return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])


# def compute_hybrid_similarity(text1: str, text2: str, emb1: np.ndarray, emb2: np.ndarray, alpha: float = 0.5, beta: float = 0.5):
#     semantic = float(cosine_similarity([emb1], [emb2])[0][0])
#     lexical = float(compute_lexical_similarity(text1, text2))
#     hybrid = float(alpha * semantic + beta * lexical)
#     return hybrid, semantic, lexical


from __future__ import annotations
from difflib import SequenceMatcher
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pythainlp.tokenize import word_tokenize


def remove_template_text(student_text: str, template_text: str, threshold=0.9) -> str:
    """Removes lines from student_text that are highly similar to lines in template_text."""
    if not template_text:
        return student_text

    student_lines = student_text.strip().split('\n')
    template_lines = template_text.strip().split('\n')
    result_lines = []

    for s_line in student_lines:
        is_template_line = any(
            SequenceMatcher(None, s_line.strip(),
                            t_line.strip()).ratio() > threshold
            for t_line in template_lines
        )
        if not is_template_line:
            result_lines.append(s_line)

    return "\n".join(result_lines).strip()


def preprocess_text(text: str) -> str:
    """Cleans and tokenizes text for lexical analysis."""
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace
    tokens = word_tokenize(text, engine="newmm")
    return " ".join(tokens)


def compute_lexical_similarity(text1: str, text2: str, ngram_range=(1, 3)) -> float:
    """Computes TF-IDF cosine similarity between two texts."""
    if not text1 or not text2:
        return 0.0

    t1, t2 = preprocess_text(text1), preprocess_text(text2)
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range)
    try:
        tfidf = vectorizer.fit_transform([t1, t2])
        return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    except ValueError:
        # Occurs if one or both texts are empty after preprocessing
        return 0.0


def compute_hybrid_similarity(
    text1: str, text2: str,
    emb1: np.ndarray, emb2: np.ndarray,
    alpha: float = 0.5, beta: float = 0.5
) -> tuple[float, float, float]:
    """
    Computes a hybrid similarity score based on semantic and lexical similarity.
    Returns: (hybrid_score, semantic_score, lexical_score)
    """
    if emb1 is None or emb2 is None:
        semantic = 0.0
    else:
        semantic = cosine_similarity([emb1], [emb2])[0][0]

    lexical = compute_lexical_similarity(text1, text2)

    hybrid = (alpha * semantic) + (beta * lexical)

    return float(hybrid), float(semantic), float(lexical)
