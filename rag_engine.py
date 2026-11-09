"""Simple retrieval-based answers (RAG retrieval step) over a local knowledge base.

Documents in the knowledge/ folder are split into passages, indexed with TF-IDF,
and the passages most similar to the question are returned with their sources.
The retrieved passages can also be passed to an LLM as context (see get_context).
"""
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
MIN_SCORE = 0.08  # below this similarity, we say we don't know
NO_INFO = ("I don't have reliable information on that in my knowledge base. "
           "Please consult a doctor.")

# Short forms people type that don't appear in the documents verbatim
SYNONYMS = {
    "bp": "blood pressure bp",
    "sugar": "blood sugar glucose",
    "weight": "weight bmi",
    "exercise": "physical activity exercise",
}


def _load_passages():
    """Return a list of (passage_text, source_file) from the knowledge folder."""
    passages = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for block in text.split("\n\n"):
            block = block.strip()
            # skip the "Source:" header line and very short fragments
            if block and not block.startswith("Source:") and len(block) > 40:
                passages.append((block, path.stem))
    return passages


_passages = _load_passages()
_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
_matrix = (_vectorizer.fit_transform([p[0] for p in _passages])
           if _passages else None)


def _expand(query):
    words = [SYNONYMS.get(w, w) for w in query.lower().split()]
    return " ".join(words)


def get_context(query, top_k=3):
    """Return the top matching passages as dicts: text, source, score."""
    if _matrix is None or not query.strip():
        return []
    q_vec = _vectorizer.transform([_expand(query)])
    scores = cosine_similarity(q_vec, _matrix)[0]
    ranked = scores.argsort()[::-1][:top_k]
    return [
        {"text": _passages[i][0], "source": _passages[i][1], "score": float(scores[i])}
        for i in ranked if scores[i] >= MIN_SCORE
    ]


def get_answer(query, top_k=2):
    """Return a text answer built from retrieved passages, with sources."""
    hits = get_context(query, top_k=top_k)
    if not hits:
        return NO_INFO
    parts = [h["text"] for h in hits]
    sources = ", ".join(sorted({h["source"].replace("_", " ") for h in hits}))
    return "\n\n".join(parts) + f"\n\n(Sources: {sources})"