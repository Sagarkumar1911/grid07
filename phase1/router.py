"""
Phase 1: Vector-Based Persona Matching (The Router)
====================================================
Uses sentence-transformers for local embeddings + FAISS for vector similarity.
Routes incoming posts to only the bots whose personas are relevant.
"""

import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Bot Persona Definitions
# ---------------------------------------------------------------------------

BOT_PERSONAS = {
    "bot_a": {
        "name": "Tech Maximalist",
        "description": (
            "I believe AI and crypto will solve all human problems. I am highly optimistic "
            "about technology, Elon Musk, and space exploration. I dismiss regulatory concerns."
        ),
    },
    "bot_b": {
        "name": "Doomer / Skeptic",
        "description": (
            "I believe late-stage capitalism and tech monopolies are destroying society. "
            "I am highly critical of AI, social media, and billionaires. I value privacy and nature."
        ),
    },
    "bot_c": {
        "name": "Finance Bro",
        "description": (
            "I strictly care about markets, interest rates, trading algorithms, and making money. "
            "I speak in finance jargon and view everything through the lens of ROI."
        ),
    },
}

# ---------------------------------------------------------------------------
# 2. Build the in-memory FAISS vector store
# ---------------------------------------------------------------------------

def build_persona_index(model: SentenceTransformer) -> tuple[faiss.IndexFlatIP, list[str]]:
    """
    Embeds all bot persona descriptions and loads them into a FAISS inner-product
    index (cosine similarity after L2 normalisation).

    Returns
    -------
    index   : FAISS index ready for similarity search
    bot_ids : ordered list of bot IDs (parallel to index rows)
    """
    bot_ids = list(BOT_PERSONAS.keys())
    descriptions = [BOT_PERSONAS[bid]["description"] for bid in bot_ids]

    # Encode & L2-normalise so inner product == cosine similarity
    embeddings = model.encode(descriptions, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product index
    index.add(embeddings)

    logger.info("Persona index built — %d vectors, dim=%d", index.ntotal, dim)
    return index, bot_ids


# ---------------------------------------------------------------------------
# 3. Routing function
# ---------------------------------------------------------------------------

def route_post_to_bots(
    post_content: str,
    model: SentenceTransformer,
    index: faiss.IndexFlatIP,
    bot_ids: list[str],
    threshold: float = 0.85,
) -> list[dict]:
    """
    Embed *post_content* and return every bot whose persona cosine-similarity
    score exceeds *threshold*.

    NOTE: all-MiniLM-L6-v2 scores typically fall in the 0.25–0.55 range for
    semantically related text. If using this model, set threshold=0.30.
    The 0.85 default matches the assignment spec and works with larger models
    (e.g. text-embedding-3-small). Tune per embedding model as instructed.

    Parameters
    ----------
    post_content : incoming social-media post text
    model        : shared SentenceTransformer instance
    index        : pre-built FAISS persona index
    bot_ids      : ordered list matching the index rows
    threshold    : minimum cosine similarity (0-1) to include a bot

    Returns
    -------
    List of dicts with keys: bot_id, name, score
    """
    # Encode the post (normalised → inner product == cosine similarity)
    post_vec = model.encode([post_content], normalize_embeddings=True)
    post_vec = np.array(post_vec, dtype="float32")

    # Query ALL persona vectors at once
    scores, indices = index.search(post_vec, len(bot_ids))

    matched = []
    for score, idx in zip(scores[0], indices[0]):
        if score >= threshold:
            bot_id = bot_ids[idx]
            matched.append({
                "bot_id": bot_id,
                "name": BOT_PERSONAS[bot_id]["name"],
                "score": round(float(score), 4),
            })

    # Sort by score descending for readability
    matched.sort(key=lambda x: x["score"], reverse=True)
    return matched