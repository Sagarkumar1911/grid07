"""
Phase 1: Vector-Based Persona Matching (The Router)
====================================================
Uses sentence-transformers for local embeddings + FAISS for vector similarity.
Routes incoming posts to only the bots whose personas are relevant.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

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
    index = faiss.IndexFlatIP(dim)   # Inner Product index
    index.add(embeddings)

    print(f"[Phase 1] Persona index built — {index.ntotal} vectors, dim={dim}")
    return index, bot_ids


# ---------------------------------------------------------------------------
# 3. Routing function
# ---------------------------------------------------------------------------

def route_post_to_bots(
    post_content: str,
    model: SentenceTransformer,
    index: faiss.IndexFlatIP,
    bot_ids: list[str],
    threshold: float = 0.30,   # lower threshold for demo; tune per embedding model
) -> list[dict]:
    """
    Embed *post_content* and return every bot whose persona cosine-similarity
    score exceeds *threshold*.

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


# ---------------------------------------------------------------------------
# 4. Demo / smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Phase 1 — Vector Persona Router")
    print("=" * 60)

    # Load model once; re-used across all calls
    print("\n[Phase 1] Loading embedding model …")
    model = SentenceTransformer("all-MiniLM-L6-v2")   # fast, good quality, free

    index, bot_ids = build_persona_index(model)

    test_posts = [
        "OpenAI just released a new model that might replace junior developers.",
        "Bitcoin hits new all-time high amid regulatory ETF approvals.",
        "Big Tech companies are harvesting your data and selling it to advertisers.",
        "The Fed just raised interest rates by 25 bps — bond yields are spiking.",
    ]

    THRESHOLD = 0.30   # cosine sim threshold

    for post in test_posts:
        print(f"\n📨 Post: \"{post}\"")
        results = route_post_to_bots(post, model, index, bot_ids, threshold=THRESHOLD)

        if results:
            print(f"   Matched bots (threshold={THRESHOLD}):")
            for r in results:
                bar = "█" * int(r["score"] * 30)
                print(f"   • {r['bot_id']} ({r['name']})  score={r['score']}  {bar}")
        else:
            print("   No bots matched above threshold.")