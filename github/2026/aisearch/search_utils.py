import os
import json
import requests
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

ES_HOST = os.getenv("ES_HOST")
ES_API_KEY = os.getenv("ES_API_KEY")
ES_INDEX = os.getenv("ES_INDEX", "rijksoverheid-qa-v3")
OLLAMA_URL = "http://localhost:11434"

DUTCH_STOPWORDS = {
    "de", "het", "een", "en", "van", "in", "is", "dat", "op", "te",
    "voor", "met", "aan", "er", "zijn", "wordt", "ik", "hoe", "wat",
    "wie", "waar", "wanneer", "waarom", "kan", "mag", "moet", "wil",
    "mijn", "uw", "zijn", "haar", "ons", "hun", "dit", "die", "deze"
}


@st.cache_resource
def get_es():
    return Elasticsearch(ES_HOST, api_key=ES_API_KEY)


def is_dutch(text: str) -> bool:
    words = set(text.lower().split())
    return bool(words & DUTCH_STOPWORDS)


def translate_to_dutch(text: str) -> str:
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": "qwen3:0.6b",
                "prompt": (
                    f"Translate the following text to Dutch. "
                    f"Return ONLY the Dutch translation, nothing else. "
                    f"No thinking, no explanation.\n\nText: {text}\n\nDutch:"
                ),
                "stream": False,
                "options": {"num_ctx": 512, "temperature": 0}
            },
            timeout=15
        )
        result = resp.json().get("response", text).strip()
        # strip any <think>...</think> tags qwen3 sometimes emits
        if "<think>" in result:
            result = result.split("</think>")[-1].strip()
        return result if result else text
    except Exception:
        return text


def resolve_query(query: str) -> tuple[str, str | None]:
    """Returns (query_to_use, translated_nl) — translated_nl is None if already Dutch."""
    if is_dutch(query):
        return query, None
    translated = translate_to_dutch(query)
    return translated, translated


def _bm25_retriever(query, fields):
    return {
        "standard": {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": fields,
                    "type": "best_fields"
                }
            }
        }
    }


def _semantic_retriever(query):
    return {
        "standard": {
            "query": {"semantic": {"field": "body_semantic", "query": query}}
        }
    }


def run_search(query, mode, use_title, title_boost, use_body, body_boost,
               use_semantic, rrf_window, rrf_constant, result_size):
    es = get_es()
    nl_query, translated = resolve_query(query)

    bm25_fields = []
    if use_title:
        bm25_fields.append(f"title^{title_boost}")
    if use_body:
        bm25_fields.append(f"body^{body_boost}")

    bm25_ret = _bm25_retriever(nl_query, bm25_fields) if bm25_fields else None
    sem_ret = _semantic_retriever(nl_query) if use_semantic else None

    source = ["url", "title", "body"]

    if mode == "Hybrid (RRF)":
        retrievers = [r for r in [bm25_ret, sem_ret] if r]
        if not retrievers:
            return None, translated
        body = (
            {"retriever": retrievers[0], "_source": source, "size": result_size}
            if len(retrievers) == 1
            else {
                "retriever": {
                    "rrf": {
                        "retrievers": retrievers,
                        "rank_window_size": rrf_window,
                        "rank_constant": rrf_constant
                    }
                },
                "_source": source,
                "size": result_size
            }
        )
    elif mode == "BM25 Only":
        if not bm25_fields:
            return None, translated
        body = {
            "query": {"multi_match": {"query": nl_query, "fields": bm25_fields, "type": "best_fields"}},
            "_source": source,
            "size": result_size
        }
    else:  # Semantic Only
        if not use_semantic:
            return None, translated
        body = {
            "query": {"semantic": {"field": "body_semantic", "query": nl_query}},
            "_source": source,
            "size": result_size
        }

    return es.search(index=ES_INDEX, body=body), translated


def hybrid_search(query, top_k=5):
    es = get_es()
    nl_query, translated = resolve_query(query)
    resp = es.search(index=ES_INDEX, body={
        "retriever": {
            "rrf": {
                "retrievers": [
                    _bm25_retriever(nl_query, ["title^2", "body_clean"]),
                    _semantic_retriever(nl_query)
                ],
                "rank_window_size": 50,
                "rank_constant": 20
            }
        },
        "_source": ["url", "title", "body_clean"],
        "size": top_k
    })
    return resp, translated


def hybrid_search_with_chunks(query, top_k=5, chunks_per_doc=3):
    """
    Two-pass RAG retrieval:
    Pass 1 — RRF hybrid search for best-ranked documents.
    Pass 2 — semantic inner_hits on those docs to extract the actual
              matching chunk text (not just the beginning of body).

    Returns (rrf_hits, chunk_map, translated)
      rrf_hits  : list of hit dicts from pass 1 (for source display)
      chunk_map : {url: [chunk_text, ...]} with matched chunks per doc
      translated: Dutch translation if query was English, else None
    """
    es = get_es()
    nl_query, translated = resolve_query(query)

    # Pass 1: RRF ranking
    rrf_resp = es.search(index=ES_INDEX, body={
        "retriever": {
            "rrf": {
                "retrievers": [
                    _bm25_retriever(nl_query, ["title^2", "body_clean"]),
                    _semantic_retriever(nl_query)
                ],
                "rank_window_size": 50,
                "rank_constant": 20
            }
        },
        "_source": ["url", "title", "body_clean"],
        "size": top_k
    })

    rrf_hits = rrf_resp["hits"]["hits"]
    if not rrf_hits:
        return rrf_hits, {}, translated

    doc_urls = [h["_source"]["url"] for h in rrf_hits]

    # Pass 2: semantic with inner_hits, filtered to ranked docs only
    try:
        chunk_resp = es.search(index=ES_INDEX, body={
            "query": {
                "bool": {
                    "must": {
                        "semantic": {
                            "field": "body_semantic",
                            "query": nl_query,
                            "inner_hits": {"size": chunks_per_doc}
                        }
                    },
                    "filter": {"terms": {"url": doc_urls}}
                }
            },
            "_source": ["url", "title"],
            "size": top_k
        })

        chunk_map = {}
        for h in chunk_resp["hits"]["hits"]:
            url = h["_source"]["url"]
            inner_hits = (
                h.get("inner_hits", {})
                 .get("body_semantic", {})
                 .get("hits", {})
                 .get("hits", [])
            )
            chunks = [
                c.get("_source", {}).get("text", "")
                for c in inner_hits
                if c.get("_source", {}).get("text")
            ]
            if chunks:
                chunk_map[url] = chunks

    except Exception:
        # inner_hits failed — fall back to body_clean prefix per doc
        chunk_map = {}

    return rrf_hits, chunk_map, translated
