import os, re, time, textwrap, pickle
from typing import List, Sequence, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import requests

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =========================================================
# 1) --- TUS FUNCIONES DE RECUPERACIÓN (sin cambiar) ---
# =========================================================
import faiss
from sentence_transformers import SentenceTransformer

# ---- Rutas (ajústalas o usa variables de entorno) ----
PKL_MIN_PATH = os.environ.get("PKL_MIN_PATH", "embeddings_meta_min.pkl")
FAISS_PATH   = os.environ.get("FAISS_PATH", "faiss_index_ip.bin")
SCOPUS_CSV   = os.environ.get("SCOPUS_CSV", "scopusdata.csv")
SCOPUS_SEP   = os.environ.get("SCOPUS_SEP", "|")

# ---- Caches simples ----
_model_cache = None
_meta_min_cache = None
_index_cache = None
_scopus_cache = None

# === Detección universal de idioma (fastText, sin fallback) ===
import fasttext
LID_MODEL_PATH = os.environ.get("LID_MODEL_PATH", "lid.176.ftz")
LID_NO_AUTO_DOWNLOAD = os.environ.get("LID_NO_AUTO_DOWNLOAD", "0") == "1"
LID_DOWNLOAD_URL = os.environ.get(
    "LID_DOWNLOAD_URL",
    "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
)

_lid_model_cache = None

def _get_lid_model():
    global _lid_model_cache
    if _lid_model_cache is None:
        _lid_model_cache = fasttext.load_model(LID_MODEL_PATH)
        print(f"[INFO] fastText LID cargado desde: {LID_MODEL_PATH}")
    return _lid_model_cache

def detect_lang_any(text: str) -> Tuple[str, float]:
    """
    Devuelve (iso639_1, confidence) p.ej., ('es', 0.98).
    Sin fallback: si el código no es válido, lanza excepción.
    """
    s = _safe_str(text).strip()
    if not s:
        raise ValueError("Texto vacío para detección de idioma.")
    model = _get_lid_model()
    labels, probs = model.predict(s.replace("\n", " ")[:4000], k=1)
    lab = _safe_str(labels[0]).replace("__label__", "").lower()
    conf = float(probs[0])

    # Normaliza a ISO-639-1 de 2 letras
    iso = lab.split("-")[0]
    if len(iso) > 2:
        iso = iso[:2]
    if not re.fullmatch(r"[a-z]{2}", iso):
        raise ValueError(f"Etiqueta de idioma inválida devuelta por fastText: {lab}")
    return iso, conf

def load_pkl_and_model(emb_max_seq_len=300):
    global _model_cache, _meta_min_cache
    if _model_cache is not None and _meta_min_cache is not None:
        return _model_cache, _meta_min_cache
    with open(PKL_MIN_PATH, "rb") as f:
        pkl = pickle.load(f)

    meta_min = pkl["meta_min"].copy()  # DataFrame: vec_id, chunk_uid, doc_id, chunk_id, (scopus_id), start/end
    _meta_min_cache = meta_min

    model_name = pkl.get("model", "intfloat/multilingual-e5-small")
    model = SentenceTransformer(model_name, device="cpu")
    model.max_seq_length = min(int(emb_max_seq_len), 512)
    _model_cache = model

    print(f"[INFO] Modelo: {model_name} | max_seq_length={model.max_seq_length}")
    print(f"[INFO] meta_min columnas: {list(meta_min.columns)} | filas={len(meta_min)}")
    return _model_cache, _meta_min_cache

def load_faiss():
    global _index_cache
    if _index_cache is None:
        _index_cache = faiss.read_index(FAISS_PATH)
        print(f"[INFO] Índice FAISS cargado: ntotal={_index_cache.ntotal}")
    return _index_cache

def load_scopus_csv():
    global _scopus_cache
    if _scopus_cache is None:
        df = pd.read_csv(SCOPUS_CSV, sep=SCOPUS_SEP)
        if "scopus_id" not in df.columns:
            raise ValueError(f"{SCOPUS_CSV} no tiene columna 'scopus_id'")
        df["scopus_id"] = df["scopus_id"].astype(str)
        _scopus_cache = df
        print(f"[INFO] scoupusdata.csv: filas={len(df)} | cols={len(df.columns)}")
    return _scopus_cache

def e5_encode_query(model, query_text: str):
    return model.encode([f"query: {query_text}"],
                        normalize_embeddings=True,
                        convert_to_numpy=True).astype("float32")

def search_min(query_text: str, topk: int = 5) -> pd.DataFrame:
    """
    Devuelve SOLO el meta mínimo del PKL (sin CSV):
    vec_id, score, chunk_uid, doc_id, chunk_id, (scopus_id si existe), start/end
    """
    model, meta_min = load_pkl_and_model()
    index = load_faiss()

    q = e5_encode_query(model, query_text)
    D, I = index.search(q, topk)
    vec_ids = I[0].tolist()

    hits = meta_min.set_index("vec_id").loc[vec_ids].reset_index()
    hits.insert(1, "score", D[0])

    cols_front = [c for c in ["vec_id","score","chunk_uid","doc_id","chunk_id","scopus_id","start_token","end_token"] if c in hits.columns]
    rest = [c for c in hits.columns if c not in cols_front]
    return hits[cols_front + rest].reset_index(drop=True)

def search_full_scopus(query_text: str, topk: int = 5) -> pd.DataFrame:
    """
    Une el TOP-K con TODAS las columnas de scoupusdata.csv por scopus_id.
    """
    model, meta_min = load_pkl_and_model()
    index = load_faiss()
    sc = load_scopus_csv()

    q = e5_encode_query(model, query_text)
    D, I = index.search(q, topk)
    vec_ids = I[0].tolist()

    hits = meta_min.set_index("vec_id").loc[vec_ids].reset_index()
    hits.insert(1, "score", D[0])

    if "scopus_id" not in hits.columns:
        raise ValueError("meta_min en PKL no contiene 'scopus_id'; no puedo unir con el CSV.")

    out = hits.merge(sc, how="left", on="scopus_id")

    # Orden: primero claves/score/offsets, luego TODO el CSV
    front = [c for c in ["vec_id","score","chunk_uid","doc_id","chunk_id","scopus_id","start_token","end_token"] if c in out.columns]
    csv_cols = [c for c in sc.columns if c not in front]
    return out[front + csv_cols].reset_index(drop=True)

# =========================================================
# 2) --- RE-RANKING + RAG + LLM (Ollama) ---
# =========================================================

# Re-ranking
from sentence_transformers import CrossEncoder
CROSS_ENCODER_MODEL = os.environ.get("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CE_BATCH_SIZE       = int(os.environ.get("CE_BATCH_SIZE", "64"))
W_CE, W_DENSE       = float(os.environ.get("W_CE", "0.7")), float(os.environ.get("W_DENSE", "0.3"))

# Texto prioridad para CE
TEXT_COLS = ["title","abstract","chunk_text","summary","authkeywords","keywords"]

def _safe_str(x) -> str:
    if isinstance(x, str): return x
    if x is None: return ""
    if isinstance(x, float) and x != x: return ""  # NaN
    try:
        return str(x)
    except Exception:
        return ""

def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mn, mx = float(np.nanmin(x)), float(np.nanmax(x))
    if not np.isfinite(mn) or not np.isfinite(mx) or (mx - mn) <= 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn + 1e-12)

def _first_nonempty(row: pd.Series, cols: Sequence[str]) -> str:
    for c in cols:
        if c in row and isinstance(row[c], str) and row[c].strip():
            return row[c]
    parts = []
    for c in row.index:
        name = c.lower()
        if any(tok in name for tok in ("title","abstract","summary","keywords","chunk","desc","text")):
            v = row[c]
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
    return " ".join(parts)[:4096]

_ce_cache: Optional[CrossEncoder] = None
def get_cross_encoder(model_name: str = CROSS_ENCODER_MODEL) -> CrossEncoder:
    global _ce_cache
    if _ce_cache is None:
        _ce_cache = CrossEncoder(model_name, device="cpu")
        print(f"[INFO] Cross-Encoder cargado: {model_name}")
    return _ce_cache

def _build_pairs(query_text: str, df_topk: pd.DataFrame, text_cols: Optional[List[str]]) -> Tuple[List[Tuple[str,str]], List[int]]:
    cols = text_cols or TEXT_COLS
    pairs, idx_map = [], []
    for i, row in df_topk.iterrows():
        txt = _first_nonempty(row, cols)
        pairs.append((query_text, txt if isinstance(txt, str) else ""))
        idx_map.append(i)
    return pairs, idx_map

def rerank_with_cross_encoder(query_text: str,
                              df_topk: pd.DataFrame,
                              text_cols: Optional[List[str]] = None,
                              score_dense_col: str = "score",
                              fuse_with_dense: bool = True,
                              batch_size: int = CE_BATCH_SIZE,
                              model_name: str = CROSS_ENCODER_MODEL) -> pd.DataFrame:
    if df_topk is None or len(df_topk) == 0:
        raise ValueError("df_topk vacío.")
    ce = get_cross_encoder(model_name)
    pairs, idx_map = _build_pairs(query_text, df_topk, text_cols)

    scores_ce = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start+batch_size]
        s = ce.predict(batch)
        scores_ce.append(np.asarray(s, dtype=np.float32))
    scores_ce = np.concatenate(scores_ce, axis=0) if scores_ce else np.zeros(len(df_topk), dtype=np.float32)

    out = df_topk.copy()
    out.loc[idx_map, "score_ce"] = scores_ce
    ce_norm = _minmax(out["score_ce"].values)

    if fuse_with_dense and score_dense_col in out.columns:
        dense_norm = _minmax(out[score_dense_col].values)
        out["score_dense_norm"] = dense_norm
        out["score_final"] = W_CE * ce_norm + W_DENSE * dense_norm
    else:
        out["score_final"] = ce_norm

    out = out.sort_values("score_final", ascending=False).reset_index(drop=True)
    return out

# --- RAG blocks ---
TOP_CONTEXT       = int(os.environ.get("RAG_TOP_CONTEXT", "6"))
MAX_INPUT_CHARS   = int(os.environ.get("RAG_MAX_INPUT_CHARS", "7000"))
MAX_CHUNK_CHARS   = int(os.environ.get("RAG_MAX_CHUNK_CHARS", "900"))
DO_TRIM_ABSTRACT  = os.environ.get("RAG_TRIM_ABSTRACT", "1") == "1"

def _shorten(txt, lim: int) -> str:
    s = _safe_str(txt)
    s = re.sub(r"\s+", " ", s).strip()
    return (s[:lim-3] + "...") if len(s) > lim else s

def _split_authors(raw: str) -> List[str]:
    s = _safe_str(raw)
    if not s.strip(): return []
    s = s.replace("|", ";").replace(" and ", ";")
    parts = [p.strip() for p in s.split(";") if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts

def _last_name(name: str) -> str:
    n = _safe_str(name).strip()
    if not n: return ""
    if "," in n: return n.split(",")[0].strip()
    tokens = n.split()
    return tokens[-1].strip() if tokens else n

def _format_authors_for_mention(raw: str, max_names: int = 2) -> Optional[str]:
    authors = _split_authors(raw)
    if not authors: return None
    last_names = [_last_name(a) for a in authors if _last_name(a)]
    if not last_names: return None
    if len(last_names) == 1: return last_names[0]
    if len(last_names) == 2: return f"{last_names[0]} y {last_names[1]}"
    return f"{last_names[0]} et al."

def _extract_year(row: pd.Series) -> Optional[str]:
    for c in ["year", "publication_year", "cover_date", "date"]:
        val = _safe_str(row.get(c))
        m = re.search(r"(19|20)\d{2}", val)
        if m: return m.group(0)
    return None

def build_context_blocks(df_reranked: pd.DataFrame,
                         top_k: int = TOP_CONTEXT,
                         max_chunk_chars: int = MAX_CHUNK_CHARS) -> List[Dict]:
    if df_reranked is None or len(df_reranked) == 0:
        raise ValueError("df_reranked está vacío.")
    cols_title = [c for c in ["title","chunk_title"] if c in df_reranked.columns] or ["title"]
    cols_abs   = [c for c in ["abstract","chunk_text","summary"] if c in df_reranked.columns] or ["abstract"]
    blocks = []
    for i in range(min(top_k, len(df_reranked))):
        row = df_reranked.iloc[i]
        title  = _first_nonempty(row, cols_title) or "Sin título"
        body   = _first_nonempty(row, cols_abs)
        if DO_TRIM_ABSTRACT: body = _shorten(body, max_chunk_chars)
        authors_raw = _safe_str(row.get("authors", ""))
        year  = _extract_year(row)
        doi_raw = _safe_str(row.get("doi", ""))
        blocks.append({
            "cite_id": str(row.get("scopus_id") or row.get("vec_id") or f"row{i}"),
            "title": title,
            "text": body,
            "authors_mention": _format_authors_for_mention(authors_raw),
            "authors_raw": authors_raw,
            "year": year,
            "doi_raw": doi_raw
        })
    return blocks

def _doi_url(doi_raw: str) -> str:
    doi = _safe_str(doi_raw).strip()
    if not doi: return "s/d"
    return doi if doi.lower().startswith("http") else f"https://doi.org/{doi}"

def _authors_cite_line(raw_authors: str) -> str:
    names = _split_authors(_safe_str(raw_authors))
    if not names: return "Autor(es) no disponibles"
    return "; ".join([_safe_str(n).strip() for n in names if _safe_str(n).strip()])

def render_fuentes_from_blocks(blocks: List[Dict]) -> str:
    lines = []
    for i, b in enumerate(blocks, start=1):
        autores = _authors_cite_line(b.get("authors_raw", ""))
        titulo  = _safe_str(b.get("title", "Sin título"))
        doiurl  = _doi_url(b.get("doi_raw", ""))
        lines.append(f"[{i}] {autores}; \"{titulo}\"; {doiurl}")
    return "\n".join(lines)

def compose_prompt(query: str,
                   blocks: List[Dict],
                   max_chars: int = MAX_INPUT_CHARS,
                   target_iso: str = "es") -> str:
    """
    Universal compose_prompt (English instructions) that forces the model
    to answer entirely in the language specified by target_iso (ISO 639-1).
    It keeps your [n] citation discipline and RAG constraints.
    """
    if not re.fullmatch(r"[a-z]{2}", _safe_str(target_iso).lower()):
        target_iso = "es"
    target_iso = target_iso.lower()

    header = (
        "You are an academic assistant for Retrieval-Augmented Generation (RAG).\n"
        f"Write the ENTIRE answer in the language whose ISO 639-1 code is '{target_iso}'. "
        "Do not mix languages. Ignore the language of the user message if it differs from this code.\n\n"
        "Style & constraints:\n"
        "- Be clear, concise, and evidence-based.\n"
        "- Follow APA 7th in-text style.\n"
        "- Use ONLY the provided excerpts. Do NOT invent or add external facts.\n"
        "- Every paragraph must include an explicit author mention written in the target language "
        "(e.g., 'según <autor>' or 'according to <author>') followed by the bracketed citation [n].\n"
        "- If no author is available, use 'according to source [n]' (translated into the target language if applicable).\n"
        "- Do NOT print any 'References' section at the end.\n\n"
        f"Question:\n{query}\n\n"
        "Source excerpts (with author/year if available):\n"
    )

    parts = []
    for i, b in enumerate(blocks, start=1):
        author_mention = b.get("authors_mention") or f"source [{i}]"
        year = f" ({b['year']})" if b.get("year") else ""
        head = f"[{i}] { _safe_str(b.get('title','Untitled')) } — Authors: {author_mention}{year}"
        parts.append(f"{head}\n{_safe_str(b.get('text',''))}\n")

    footer = (
        "\nWriting checklist (must obey):\n"
        f"- Use the numeric citations [1..{len(blocks)}] exactly as provided.\n"
        "- Keep paragraphs tightly focused; avoid generic boilerplate.\n"
        "- Maintain consistent tone; no bullet points unless strictly necessary.\n"
        "- Do not include any section titled 'References' or similar.\n"
    )

    fuentes_prompt = render_fuentes_from_blocks(blocks)
    prompt = header + "\n".join(parts) + footer + "\n(Do NOT print the list below)\nSources guide:\n" + fuentes_prompt
    return prompt[:max_chars]

# --- LLM (Ollama) ---
OLLAMA_HOST       = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL      = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
TEMPERATURE       = float(os.environ.get("RAG_TEMPERATURE", "0.2"))
MAX_NEW_TOKENS    = int(os.environ.get("RAG_MAX_NEW_TOKENS", "768"))
HTTP_TIMEOUT_SECS = int(os.environ.get("RAG_HTTP_TIMEOUT_SECS", "300"))

def generate_with_ollama_http(prompt: str,
                              model: str = OLLAMA_MODEL,
                              temperature: float = TEMPERATURE,
                              max_new_tokens: int = MAX_NEW_TOKENS,
                              base_url: str = OLLAMA_HOST,
                              timeout: int = HTTP_TIMEOUT_SECS,
                              target_iso: Optional[str] = None) -> str:
    """
    Sin fallback: target_iso (p.ej. 'es','en','fr') es obligatorio.
    """
    if not target_iso or not re.fullmatch(r"[a-z]{2}", target_iso):
        raise RuntimeError("target_iso requerido y debe ser un código ISO-639-1 de dos letras (ej. 'es','en').")

    system_msg = (
        f"You are an academic assistant. Write the ENTIRE answer in the language whose ISO 639-1 code is '{target_iso}'. "
        "Do not mix languages. Be concise, evidence-based, and follow APA 7th in-text citations using [n]. "
        "Use ONLY the provided excerpts."
    )

    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": prompt}
        ],
        "options": {
            "temperature": float(temperature),
            "num_predict": int(max_new_tokens),
            "stop": ["\nFuentes", "\nFUENTES", "\nReferences", "\nREFERENCIAS"]
        },
        "stream": False
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    content = _safe_str(data.get("message", {}).get("content", ""))
    if not content.strip():
        raise RuntimeError("La respuesta de Ollama está vacía.")
    return content

def extract_used_refs(answer_text: str, n_max: int) -> List[int]:
    nums = [int(m.group(1)) for m in re.finditer(r"\[(\d+)\]", _safe_str(answer_text))]
    seen, used = set(), []
    for n in nums:
        if 1 <= n <= n_max and n not in seen:
            used.append(n); seen.add(n)
    return used

def render_used_refs_report(answer_text: str, blocks: List[Dict]) -> str:
    used = extract_used_refs(answer_text, len(blocks))
    if not used:
        return "No se detectaron citas [n] en el texto."
    lines = ["Citas usadas en el texto:"]
    for n in used:
        b = blocks[n-1]
        lines.append(f" - [{n}] {_authors_cite_line(b.get('authors_raw',''))}; \"{_safe_str(b.get('title','Sin título'))}\"; {_doi_url(b.get('doi_raw',''))}")
    return "\n".join(lines)

# =========================================================
# 3) --- API (FastAPI) ---
# =========================================================
DEFAULT_TOPK = int(os.environ.get("API_TOPK", "100"))

class AskRequest(BaseModel):
    query: str
    topk: Optional[int] = None

class AskResponse(BaseModel):
    query: str
    answer_text: str
    blocks: List[Dict[str, Any]]
    used_refs_report: str
    timing_ms: Dict[str, int]

app = FastAPI(title="Centinela RAG API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en producción: restringe a tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    try:
        # Artefactos de RAG
        _ = load_pkl_and_model()
        _ = load_faiss()
        _ = load_scopus_csv()
        # Modelo de idioma (obligatorio, sin fallback)
        _ = _get_lid_model()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    q = (payload.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query vacía.")

    # --- Detección de idioma (sin fallback) ---
    try:
        detected_iso, conf = detect_lang_any(q)
        print(f"[INFO] Idioma detectado: {detected_iso} (conf={conf:.3f})")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fallo detectando idioma: {e}")

    topk = payload.topk or DEFAULT_TOPK
    t0 = time.time()

    # 1) Recuperación (denso FAISS + unión CSV)
    try:
        df_topk: pd.DataFrame = search_full_scopus(q, topk=topk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fallo en búsqueda: {e}")
    t1 = time.time()

    # 2) Re-ranking (CE + denso)
    try:
        reranked: pd.DataFrame = rerank_with_cross_encoder(
            query_text=q,
            df_topk=df_topk,
            text_cols=None,
            score_dense_col="score",
            fuse_with_dense=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fallo en re-ranking: {e}")
    t2 = time.time()

    # 3) Bloques de contexto
    try:
        blocks = build_context_blocks(reranked, top_k=TOP_CONTEXT, max_chunk_chars=MAX_CHUNK_CHARS)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fallo construyendo bloques: {e}")

    # 4) Prompt universal y 5) Generación forzada al idioma detectado
    prompt = compose_prompt(q, blocks, max_chars=MAX_INPUT_CHARS, target_iso=detected_iso)
    try:
        answer_text = generate_with_ollama_http(prompt, target_iso=detected_iso)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fallo generando respuesta LLM: {e}")
    t3 = time.time()

    # 6) Auditoría
    try:
        used_refs_report = render_used_refs_report(answer_text, blocks)
    except Exception as e:
        used_refs_report = f"(No se pudo auditar las citas) Detalle: {e}"

    timing_ms = {
        "search":   int((t1 - t0) * 1000),
        "rerank":   int((t2 - t1) * 1000),
        "generate": int((t3 - t2) * 1000),
        "total":    int((t3 - t0) * 1000),
    }

    return AskResponse(
        query=q,
        answer_text=answer_text,
        blocks=blocks,
        used_refs_report=used_refs_report,
        timing_ms=timing_ms
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
