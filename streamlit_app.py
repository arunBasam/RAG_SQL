import os
import streamlit as st
import traceback

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain


# -------------------------------
# 🔐 Page Config
# -------------------------------
st.set_page_config(
    page_title="RAG SQL Assistant",
    layout="centered"
)

st.title("📄 RAG Document Assistant")
st.caption("Retrieval-Augmented Generation using ChromaDB + OpenAI")

# -------------------------------
# 🔐 Load API Key
# -------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not set")
    st.stop()


# -------------------------------
# 📁 Load & Index Documents (cached)
# -------------------------------
@st.cache_resource
def load_vectorstore():
    data_folder = "DataFolder"
    all_chunks = []

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    for filename in os.listdir(data_folder):
        file_path = os.path.join(data_folder, filename)

        try:
            if filename.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                docs = loader.load()
            elif filename.lower().endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()
            else:
                continue

            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)

        except Exception as e:
            st.warning(f"Failed to load {filename}")
            traceback.print_exc()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings
    )

    return vectordb


with st.spinner("🔎 Indexing documents..."):
    vectordb = load_vectorstore()


# -------------------------------
# 🤖 Create RAG Chain
# -------------------------------
llm = ChatOpenAI(
    model="gpt-4-1106-preview",
    openai_api_key=OPENAI_API_KEY,
    temperature=0
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectordb.as_retriever()
)

# -------------------------------
# 💬 Chat UI
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask a question about the documents:")

if query:
    with st.spinner("Thinking..."):
        result = qa_chain.invoke({
            "question": query,
            "chat_history": st.session_state.chat_history
        })

        answer = result["answer"]
        st.session_state.chat_history.append((query, answer))

# -------------------------------
# 📜 Display Chat History
# -------------------------------
for q, a in st.session_state.chat_history[::-1]:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Assistant:** {a}")
    st.markdown("---")
