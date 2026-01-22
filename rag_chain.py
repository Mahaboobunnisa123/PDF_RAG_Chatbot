# rag_chain.py
import os
import time
from typing import List, Dict, Any, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
NOT_FOUND_MSG = "Please ask a relevant question from the uploaded PDF."

# optional reranker
try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

# optional Google / LangChain Gemini wrapper
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None

LOG_DIR = "logs"
UNANSWERED_LOG = os.path.join(LOG_DIR, "unanswered.log")
os.makedirs(LOG_DIR, exist_ok=True)


def log_unanswered(query: str):
    with open(UNANSWERED_LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {query}\n")


def _norm(a: np.ndarray):
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return a / n


def mmr_select(query_emb: np.ndarray, candidate_embs: np.ndarray, candidate_indices: List[int], top_k: int = 5, lambda_param: float = 0.6) -> List[int]:
    if candidate_embs.size == 0:
        return []
    c = _norm(candidate_embs)
    q = query_emb / (np.linalg.norm(query_emb) + 1e-12)
    sims = (c @ q.reshape(-1, 1)).squeeze()
    selected = []
    available = set(range(len(sims)))
    first = int(np.argmax(sims))
    selected.append(first)
    available.remove(first)
    while len(selected) < min(top_k, len(sims)):
        best_score = -1e9
        best_i = None
        for i in available:
            sim_q = sims[i]
            sim_sel = max(float(c[i] @ c[j].T) for j in selected) if selected else 0.0
            score = lambda_param * sim_q - (1 - lambda_param) * sim_sel
            if score > best_score:
                best_score = score
                best_i = i
        if best_i is None:
            break
        selected.append(best_i)
        available.remove(best_i)
    return [candidate_indices[i] for i in selected]


def retrieve_candidates(query: str, model: SentenceTransformer, index, metadata: List[Dict[str, Any]], top_n: int = 40) -> List[Dict[str, Any]]:
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
    if index.ntotal == 0:
        return []
    k = min(top_n, index.ntotal)
    scores, ids = index.search(q_emb, k)
    res = []
    for sc, iid in zip(scores[0].tolist(), ids[0].tolist()):
        if iid == -1:
            continue
        m = metadata[iid]
        res.append({"id": iid, "score": float(sc), "text": m.get("text"), "source": m.get("source"), "page": m.get("page")})
    return res


def build_prompt(query: str, retrieved: List[Dict[str, Any]]) -> str:
    pieces = []
    for r in retrieved:
        t = (r.get("text") or "").replace("\n", " ").strip()
        if len(t) > 900:
            t = t[:900] + "..."
        pieces.append(f"[{r.get('source')} | page {r.get('page')} | score {r.get('score'):.3f}]\n{t}")
    context = "\n\n".join(pieces)
    prompt = f"""
You are a strict document assistant. Use ONLY the CONTEXT below to answer. If the answer cannot be found in the context, reply exactly:
{NOT_FOUND_MSG}

CONTEXT:
{context}

QUESTION:
{query}

INSTRUCTIONS:
- Do not invent facts or guess.
- Keep answer concise and include short citations (source, page) if used.
"""
    return prompt.strip()


def call_gemini_strict(prompt: str, temperature: float = 0.0, model_name: str = "gemini-2.5-flash") -> str:
    """
    Try to call Google Gemini via langchain_google_genai if installed.
    If not installed or call fails, raise RuntimeError so the caller can fallback.
    """
    if ChatGoogleGenerativeAI is None:
        raise RuntimeError("LangChain Google wrapper not installed (langchain_google_genai).")
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    try:
        out = llm.invoke(prompt)
        if isinstance(out, str):
            return out.strip()
        if hasattr(out, "content"):
            return str(out.content).strip()
        return str(out).strip()
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}")


_cached = {}


def rag_answer(
    query: str,
    model: SentenceTransformer,
    index,
    metadata: List[Dict[str, Any]],
    top_k: int = 5,
    threshold: float = 0.30,
    use_reranker: bool = False,
    scanned_pdf: bool = False
) -> Tuple[str, List[Dict[str, Any]]]:
    cache_key = f"{query}|{top_k}|{threshold}|{use_reranker}"
    if cache_key in _cached:
        return _cached[cache_key]

    # retrieve pool
    candidates = retrieve_candidates(query, model, index, metadata, top_n=max(40, top_k * 8))
    if not candidates:
        log_unanswered(query)
        _cached[cache_key] = (NOT_FOUND_MSG, [])
        return _cached[cache_key]

    # quick confidence check
    top_score = candidates[0].get("score", 0.0)
    if top_score < 0.12:
        log_unanswered(query)
        _cached[cache_key] = (NOT_FOUND_MSG, candidates[:top_k])
        return _cached[cache_key]

    # optional reranker
    if use_reranker and CrossEncoder is not None:
        try:
            ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            pairs = [[query, c["text"][:512]] for c in candidates]
            rr = ce.predict(pairs)
            for c, s in zip(candidates, rr):
                c["rerank"] = float(s)
            candidates.sort(key=lambda x: x.get("rerank", x.get("score", 0.0)), reverse=True)
        except Exception as e:
            print("[rag_chain] Reranker error:", e)

    # embeddings for MMR
    cand_texts = [c["text"] for c in candidates]
    cand_embs = model.encode(cand_texts, convert_to_numpy=True)
    cand_embs = cand_embs.astype("float32")
    cand_embs = cand_embs / (np.linalg.norm(cand_embs, axis=1, keepdims=True) + 1e-12)

    q_emb = model.encode([query], convert_to_numpy=True).astype("float32").reshape(-1)
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)

    pool_size = min(40, len(candidates))
    pool_embs = cand_embs[:pool_size]
    pool_indices = [candidates[i]["id"] for i in range(pool_size)]

    selected_ids = mmr_select(q_emb, pool_embs, pool_indices, top_k=top_k, lambda_param=0.7)

    retrieved = []
    id2candidate = {c["id"]: c for c in candidates}
    for sid in selected_ids:
        c = id2candidate.get(sid)
        if c:
            retrieved.append(c)

    # best score (use rerank if present)
    best_score = candidates[0].get("rerank") if any("rerank" in c for c in candidates) else candidates[0].get("score", 0.0)
    if best_score < threshold:
        log_unanswered(query)
        if scanned_pdf:
            msg = "This PDF seems scanned (image-based). Please upload a text-based PDF (selectable text)."
            _cached[cache_key] = (msg, retrieved)
            return _cached[cache_key]
        _cached[cache_key] = (NOT_FOUND_MSG, retrieved)
        return _cached[cache_key]

    # build prompt and call LLM (try Gemini if available)
    prompt = build_prompt(query, retrieved)
    try:
        temperature = 0.0 if best_score >= 0.6 else 0.1
        # attempt Gemini if configured; default model name is one commonly available
        answer = call_gemini_strict(prompt, temperature=temperature)
    except Exception as e:
        print("[rag_chain] LLM error:", e)
        # fallback: return "I don't know." (you can add other LLM backends here)
        log_unanswered(query)
        if scanned_pdf:
           msg = "This PDF seems scanned (image-based). Please upload a text-based PDF (selectable text)."
           _cached[cache_key] = (msg, retrieved)
           return _cached[cache_key]
        _cached[cache_key] = (NOT_FOUND_MSG, retrieved)
        return _cached[cache_key]

    if not answer or "i don't know" in answer.lower():
        _cached[cache_key] = (NOT_FOUND_MSG, retrieved)
        return _cached[cache_key]

    _cached[cache_key] = (answer.strip(), retrieved)
    return _cached[cache_key]
