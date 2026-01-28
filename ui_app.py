# ui_app.py (advanced retrievers)
import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# Advanced retrievers
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
# (Alternative: EmbeddingsFilter, gzip-like; we’ll use LLM compressor for clarity)

# --------- Config / init ----------
load_dotenv()  # expects .env with OPENAI_API_KEY=sk-xxx

DATA_DIR = Path("DataFolder")      # where you drop PDFs/TXTs
PERSIST_DIR = Path("./chroma_db")  # where Chroma stores vectors
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

st.set_page_config(page_title="RAG Chat", page_icon="💬", layout="wide")
st.title("💬 RAG Chatbot")
st.caption("Ask questions grounded in your local knowledge base. Switch retriever strategies in the sidebar.")

# --------- Sidebar (controls) ----------
with st.sidebar:
    st.subheader("Ingest documents")
    uploaded = st.file_uploader("Upload PDF or TXT", accept_multiple_files=True, type=["pdf", "txt"])
    make_persist = st.checkbox("Persist after ingest", value=True)
    ingest_btn = st.button("Ingest to Knowledge Base")

    st.divider()
    st.subheader("Retriever strategy")

    strategy = st.selectbox(
        "Choose retriever",
        [
            "Vector (kNN)",
            "Vector (MMR)",
            "Multi-Query (LLM expands)",
            "BM25 (lexical)",
            "Ensemble (BM25 + Vector)"
        ],
        index=0
    )

    # Common knobs
    top_k = st.slider("k (final chunks)", min_value=2, max_value=12, value=6)
    # Vector-specific knobs
    fetch_k = st.slider("fetch_k (vector prefetch)", 10, 100, 40)
    lambda_mult = st.slider("MMR lambda (diversity vs relevance)", 0.0, 1.0, 0.5, step=0.05)

    st.divider()
    use_compression = st.checkbox("Enable contextual compression (LLM trims retrieved chunks)", value=False)

    st.caption("Tip: BM25 relies on raw text. If app restarts, re-ingest or re-load docs to rebuild BM25. Vector search loads from Chroma persistently.")

# --------- Caches ----------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=MODEL_NAME)

@st.cache_resource(show_spinner=False)
def get_vectorstore():
    emb = get_embeddings()
    vs = Chroma(persist_directory=str(PERSIST_DIR), embedding_function=emb)
    return vs

@st.cache_resource(show_spinner=False)
def get_llm():
    # Uses OPENAI_API_KEY from env; do not hard-code
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# --------- Document helpers ----------
def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)

def load_docs_from_files(files_or_none):
    """Save uploaded files and return LangChain Documents split into chunks."""
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs_all = []

    for f in files_or_none or []:
        fname = f.name
        target = DATA_DIR / fname
        target.write_bytes(f.read())

        try:
            if fname.lower().endswith(".pdf"):
                loader = PyPDFLoader(str(target))
            else:
                loader = TextLoader(str(target), encoding="utf-8")
            docs = loader.load()
            chunks = splitter.split_documents(docs)
            chunks = [c for c in chunks if c.page_content and c.page_content.strip()]
            docs_all.extend(chunks)
        except Exception as e:
            st.warning(f"Failed to process {fname}: {e}")

    return docs_all

def load_docs_from_data_dir():
    """(Re)load all docs from DATA_DIR to build BM25 even after restart."""
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    out = []
    for p in DATA_DIR.glob("*"):
        if not p.is_file():
            continue
        try:
            if p.suffix.lower() == ".pdf":
                docs = PyPDFLoader(str(p)).load()
            elif p.suffix.lower() == ".txt":
                docs = TextLoader(str(p), encoding="utf-8").load()
            else:
                continue
            chunks = splitter.split_documents(docs)
            chunks = [c for c in chunks if c.page_content and c.page_content.strip()]
            out.extend(chunks)
        except Exception as e:
            st.warning(f"Failed loading {p.name}: {e}")
    return out

def ingest_documents(files):
    docs = load_docs_from_files(files)
    if not docs:
        st.info("No text chunks found (are these scanned PDFs?).")
        return 0
    vs = get_vectorstore()
    vs.add_documents(docs)
    if make_persist:
        vs.persist()
    # stash docs for BM25 in session so Ensemble works immediately
    st.session_state.setdefault("bm25_docs", [])
    st.session_state["bm25_docs"].extend(docs)
    return len(docs)

# --------- Retriever factory ----------
def build_retriever(strategy_name: str, k: int, fetch_k: int, lambda_mult: float, compression: bool):
    llm = get_llm()
    vs = get_vectorstore()

    # Base vector retrievers
    vec_knn = vs.as_retriever(search_kwargs={"k": k})
    vec_mmr = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult}
    )

    # BM25 retriever (lexical)
    # Build from docs in memory (session) or reload from DATA_DIR
    if "bm25_docs" not in st.session_state or not st.session_state["bm25_docs"]:
        st.session_state["bm25_docs"] = load_docs_from_data_dir()
    bm25 = BM25Retriever.from_documents(st.session_state["bm25_docs"])
    bm25.k = k

    # Choose strategy
    if strategy_name == "Vector (kNN)":
        base = vec_knn
    elif strategy_name == "Vector (MMR)":
        base = vec_mmr
    elif strategy_name == "Multi-Query (LLM expands)":
        base = MultiQueryRetriever.from_llm(retriever=vec_knn, llm=llm)
    elif strategy_name == "BM25 (lexical)":
        base = bm25
    elif strategy_name == "Ensemble (BM25 + Vector)":
        # Hybrid: combine lexical and semantic
        base = EnsembleRetriever(retrievers=[bm25, vec_knn], weights=[0.5, 0.5])
    else:
        base = vec_knn  # fallback

    # Optional contextual compression
    if compression:
        compressor = LLMChainExtractor.from_llm(llm)
        base = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base)

    return base

# --------- Chain builder ----------
@st.cache_resource(show_spinner=False)
def make_chain_cached(strategy_name: str, k: int, fetch_k: int, lambda_mult: float, compression: bool):
    llm = get_llm()
    retriever = build_retriever(strategy_name, k, fetch_k, lambda_mult, compression)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return chain

def get_chain(strategy_name: str, k: int, fetch_k: int, lambda_mult: float, compression: bool):
    # Streamlit caches by value – pack args into key for cache separation
    key = f"{strategy_name}|k={k}|fk={fetch_k}|lam={lambda_mult}|cmp={compression}"
    # clear and rebuild if needed after ingest
    return make_chain_cached(strategy_name, k, fetch_k, lambda_mult, compression)

# --------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
ensure_dirs()

# --------- Ingestion ----------
if ingest_btn:
    with st.spinner("Ingesting…"):
        added = ingest_documents(uploaded)
    if added:
        st.success(f"Ingested {added} chunks.")
        # Clear caches so the new vectors/docs are used immediately
        get_vectorstore.clear()
        make_chain_cached.clear()

# --------- Sidebar: history ----------
with st.sidebar:
    st.subheader("Conversation History")
    if st.button("Clear history"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()
    for i, m in enumerate(st.session_state.messages):
        label = "🧑 You" if m["role"] == "user" else "🤖 Bot"
        st.markdown(f"**{label}:** {m['content'][:80]}{'…' if len(m['content'])>80 else ''}")

# --------- Main chat UI ----------
for m in st.session_state.messages:
    with st.chat_message("user" if m["role"] == "user" else "assistant"):
        st.write(m["content"])

user_input = st.chat_input("Ask a question about your documents…")
if user_input:
    # Show user msg
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Answer
    with st.chat_message("assistant"):
        if not os.environ.get("OPENAI_API_KEY"):
            st.error("Missing OPENAI_API_KEY. Add it to your .env file in the project root.")
        else:
            with st.spinner("Thinking…"):
                chain = get_chain(strategy, top_k, fetch_k, lambda_mult, use_compression)
                result = chain.invoke({
                    "question": user_input,
                    "chat_history": st.session_state.chat_history
                })
                answer = result["answer"]
                st.write(answer)

                # Sources
                srcs = result.get("source_documents", [])
                if srcs:
                    with st.expander("Sources"):
                        for i, d in enumerate(srcs, 1):
                            meta = d.metadata or {}
                            st.markdown(f"- **[{i}]** {meta.get('source', 'unknown')}  (page {meta.get('page','n/a')})")

                # Save to history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.chat_history.append((user_input, answer))
