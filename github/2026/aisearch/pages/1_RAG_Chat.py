import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import requests
import streamlit as st
from search_utils import hybrid_search_with_chunks

st.set_page_config(page_title="RAG Chat", page_icon="🤖", layout="wide")

OLLAMA_URL = "http://localhost:11434"

MODELS = {
    "llama3.1:8b":  "Best quality · good Dutch · 4.9 GB",
    "mistral:7b":   "Fast · decent Dutch · 4.1 GB",
    "phi3:latest":  "Smaller · limited Dutch · 2.2 GB",
    "qwen3:0.6b":   "Fastest · basic only · 522 MB",
}


def build_context(hits, chunk_map, max_chars, use_chunks_for_llm=False):
    """
    Builds LLM context from retrieved docs.
    By default sends full body_clean (best for answer quality).
    use_chunks_for_llm=True sends only matched chunks (smaller, faster).
    chunk_map still used for source display / debug regardless.
    """
    parts = []
    for i, hit in enumerate(hits, 1):
        src = hit["_source"]
        url = src.get("url", "")
        title = src.get("title", "")
        full_body = (src.get("body_clean") or src.get("body", ""))

        if use_chunks_for_llm:
            chunks = chunk_map.get(url)
            content = "\n\n".join(chunks)[:max_chars] if chunks else full_body[:max_chars]
        else:
            content = full_body[:max_chars]

        parts.append(f"[Bron {i}] {title}\nURL: {url}\n{content}")

    return "\n\n---\n\n".join(parts)


def stream_ollama(model, prompt, ctx_size):
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": True,
              "options": {"num_ctx": ctx_size, "temperature": 0}},
        stream=True,
        timeout=180
    )
    resp.raise_for_status()
    for line in resp.iter_lines():
        if line:
            chunk = json.loads(line)
            if not chunk.get("done"):
                yield chunk.get("response", "")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ RAG Settings")

    model = st.selectbox("LLM Model", list(MODELS.keys()), index=0)
    st.caption(MODELS[model])

    st.divider()

    with st.expander("🔍 Retrieval", expanded=True):
        top_k = st.slider("Context documents", 1, 10, 5,
                          help="Docs retrieved from Elastic")
        chunks_per_doc = st.slider("Matched chunks per doc", 1, 5, 3,
                                   help="How many matching chunks to extract per document (pass 2)")
        max_chars = st.slider("Max chars per context entry", 500, 4000, 2000, 250,
                              help="Cap on combined chunk text per document")

    with st.expander("🤖 Generation", expanded=True):
        ctx_size = st.select_slider(
            "LLM context window (tokens)",
            options=[2048, 4096, 8192, 16384],
            value=4096,
            help="Higher = more context but slower"
        )
        language = st.radio("Response language", ["Nederlands", "English"], index=0)

    with st.expander("🔧 Debug", expanded=False):
        show_sources = st.checkbox("Show retrieved sources", value=True)
        use_chunks_for_llm = st.checkbox(
            "Use matched chunks only (not full doc)",
            value=False,
            help="OFF = full document body sent to LLM (better answers). ON = only matched 300-tok chunks (faster, smaller context)."
        )
        show_chunks = st.checkbox("Show matched chunks", value=False,
                                  help="Shows which text chunks matched semantically")
        show_prompt = st.checkbox("Show full prompt", value=False)

    st.divider()
    st.info("👈 Use the sidebar navigation to switch pages")


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🤖 RAG Chat — Rijksoverheid")
st.caption(f"Model: `{model}` · Top-{top_k} docs · {chunks_per_doc} chunks/doc · {language}")

question = st.text_input(
    "Ask a question",
    placeholder="Hoe vraag ik een paspoort aan? / How do I apply for a passport?",
    label_visibility="collapsed"
)

if question:
    # 1. Two-pass retrieval
    with st.spinner("Retrieving relevant documents and matched chunks…"):
        try:
            hits, chunk_map, translated = hybrid_search_with_chunks(
                question, top_k=top_k, chunks_per_doc=chunks_per_doc
            )
        except Exception as e:
            st.error(f"Elasticsearch error: {e}")
            st.stop()

    if translated:
        st.info(f"🌐 Translated to Dutch: **{translated}**")

    if not hits:
        st.warning("No relevant documents found in the knowledge base.")
        st.stop()

    # Count how many docs used chunks vs fallback
    chunk_hits = sum(1 for h in hits if h["_source"].get("url") in chunk_map)
    if chunk_hits > 0:
        st.caption(f"✅ Matched chunks extracted for {chunk_hits}/{len(hits)} docs")
    else:
        st.caption("⚠️ inner_hits unavailable — using body prefix fallback")

    context = build_context(hits, chunk_map, max_chars, use_chunks_for_llm)

    # 2. Show sources
    if show_sources:
        with st.expander(f"📚 Retrieved {len(hits)} sources", expanded=False):
            for i, hit in enumerate(hits, 1):
                src = hit["_source"]
                score = hit.get("_score") or 0.0
                url = src.get("url", "")
                has_chunks = url in chunk_map
                st.markdown(
                    f"**{i}. [{src.get('title', '')}]({url})**  "
                    f"`score={score:.4f}` "
                    f"{'🧩 chunks' if has_chunks else '📄 fallback'}"
                )
                st.caption(url)
                if show_chunks and has_chunks:
                    for j, chunk in enumerate(chunk_map[url], 1):
                        st.text(f"  chunk {j}: {chunk[:200]}…")
                else:
                    st.text((src.get("body_clean") or src.get("body", ""))[:250] + "…")
                st.divider()

    # 3. Build prompt
    if language == "Nederlands":
        prompt = f"""Je bent een behulpzame assistent die vragen beantwoordt op basis van informatie van rijksoverheid.nl.

Regels:
- Gebruik ALLEEN de onderstaande bronnen.
- Begin direct met het antwoord. Herhaal de vraag NIET.
- Vermeld aan het einde welke bron(nen) je hebt gebruikt.
- Als het antwoord niet in de bronnen staat, zeg dan: "Deze informatie staat niet in de beschikbare bronnen."

Bronnen:
{context}

Vraag: {question}

Antwoord:"""
    else:
        prompt = f"""You are a helpful assistant answering questions based on information from the Dutch government website rijksoverheid.nl.

Rules:
- Use ONLY the sources below.
- Start your answer directly. Do NOT repeat the question.
- Cite which source(s) you used at the end.
- If the answer is not in the sources, say: "This information is not available in the provided sources."

Sources:
{context}

Question: {question}

Answer:"""

    if show_prompt:
        with st.expander("🔧 Full prompt sent to LLM"):
            st.code(prompt, language="text")

    # 4. Generate
    st.markdown("### Answer")
    answer_placeholder = st.empty()
    full_answer = ""

    try:
        for token in stream_ollama(model, prompt, ctx_size):
            full_answer += token
            answer_placeholder.markdown(full_answer + "▌")
        answer_placeholder.markdown(full_answer)
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to Ollama at `localhost:11434`. Run `ollama serve` first.")
    except requests.exceptions.HTTPError as e:
        st.error(f"Ollama error: {e}. Is `{model}` pulled? Run `ollama pull {model}`.")
    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.markdown("""
    **Ask any question about Dutch government topics:**
    - *Hoe vraag ik een paspoort aan?*
    - *Wat zijn de regels voor belasting aangifte?*
    - *Hoe werkt de zorgtoeslag?*
    - *What are the rules for working in the Netherlands?*

    The RAG pipeline retrieves relevant pages from `rijksoverheid.nl` and uses the selected LLM to answer.

    👈 Tune model, context size, and language in the sidebar.
    """)
