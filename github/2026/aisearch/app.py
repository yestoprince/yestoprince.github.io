import streamlit as st
from search_utils import run_search

st.set_page_config(page_title="Rijksoverheid Search", page_icon="🔍", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Search Settings")

    mode = st.radio(
        "Search Mode",
        ["Hybrid (RRF)", "BM25 Only", "Semantic Only"],
        index=0,
        help="Hybrid combines BM25 + semantic via Reciprocal Rank Fusion"
    )

    st.divider()

    bm25_active = mode in ("Hybrid (RRF)", "BM25 Only")
    with st.expander("📝 BM25 / Keyword Fields", expanded=bm25_active):
        use_title = st.checkbox("Title", value=True, disabled=not bm25_active)
        title_boost = st.slider("Title boost", 0.1, 5.0, 2.0, 0.1,
                                disabled=not (bm25_active and use_title))
        use_body = st.checkbox("Body", value=True, disabled=not bm25_active)
        body_boost = st.slider("Body boost", 0.1, 5.0, 1.0, 0.1,
                               disabled=not (bm25_active and use_body))

    st.divider()

    sem_active = mode in ("Hybrid (RRF)", "Semantic Only")
    with st.expander("🧠 Semantic Field", expanded=sem_active):
        use_semantic = st.checkbox("Body (semantic)", value=True, disabled=not sem_active)

    st.divider()

    with st.expander("⚖️ RRF Settings", expanded=(mode == "Hybrid (RRF)")):
        rrf_window = st.slider("Rank window size", 10, 200, 50, 10,
                               disabled=(mode != "Hybrid (RRF)"),
                               help="Candidates per retriever before fusion")
        rrf_constant = st.slider("Rank constant (k)", 1, 100, 20, 1,
                                 disabled=(mode != "Hybrid (RRF)"),
                                 help="Higher = flatter score distribution")

    st.divider()
    result_size = st.slider("Results to return", 5, 50, 10, 5)

    st.divider()
    st.info("👈 Use the sidebar navigation to switch pages")

# ── Main ─────────────────────────────────────────────────────────────────────
st.title("🔍 Rijksoverheid Search")
st.caption(f"Index: `rijksoverheid-qa-v3` · Mode: **{mode}**")

query = st.text_input("Search", placeholder="bijv. belasting aangifte, paspoort aanvragen …",
                      label_visibility="collapsed")

if query:
    with st.spinner("Searching…"):
        resp, translated = run_search(
            query, mode,
            use_title, title_boost,
            use_body, body_boost,
            use_semantic,
            rrf_window, rrf_constant,
            result_size
        )

    if resp is None:
        st.warning("Enable at least one field.")
        st.stop()

    if translated:
        st.info(f"🌐 Translated to Dutch: **{translated}**")

    hits = resp["hits"]["hits"]
    total = resp["hits"]["total"]["value"]
    st.markdown(f"**{total} results** — showing top {len(hits)}")
    st.divider()

    if not hits:
        st.info("No results found.")
    else:
        for i, hit in enumerate(hits, 1):
            src = hit["_source"]
            score = hit.get("_score") or 0.0
            title = src.get("title") or src.get("url", "")
            url = src.get("url", "")
            body = src.get("body", "")
            preview = body[:400].rsplit(" ", 1)[0] + "…" if len(body) > 400 else body

            with st.container():
                col1, col2 = st.columns([9, 1])
                with col1:
                    st.markdown(f"#### {i}. [{title}]({url})")
                    st.caption(f"🔗 {url}")
                with col2:
                    st.metric("Score", f"{score:.4f}")
                st.markdown(preview)
                with st.expander("Full body"):
                    st.write(body)
                st.divider()

else:
    st.markdown("""
    **How to use:**
    - Type keywords in Dutch or English
    - Tune search mode and field weights in the sidebar
    - **Hybrid (RRF)** — best results combining keyword + semantic
    - **BM25 Only** — exact keyword matching with field boosts
    - **Semantic Only** — meaning-based via multilingual-e5-small

    👈 Use **Switch to RAG Chat** in the sidebar to ask natural language questions.
    """)
