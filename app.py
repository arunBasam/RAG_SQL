import os
import traceback

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_openai import ChatOpenAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain


# -------------------------------
# 🔐 Load OpenAI API Key safely
# -------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY not found. Please set it as an environment variable."
    )


# -------------------------------
# 📁 Load & split documents
# -------------------------------
DATA_FOLDER = "DataFolder"
all_chunks = []

text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0
)

for filename in os.listdir(DATA_FOLDER):
    file_path = os.path.join(DATA_FOLDER, filename)

    try:
        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            chunks = text_splitter.split_documents(documents)
            all_chunks.extend(chunks)
            print(f"Loaded PDF: {filename}, chunks: {len(chunks)}")

        elif filename.lower().endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()
            chunks = text_splitter.split_documents(documents)
            all_chunks.extend(chunks)
            print(f"Loaded TXT: {filename}, chunks: {len(chunks)}")

    except Exception as e:
        print(f"Error loading file {filename}: {e}")
        traceback.print_exc()

print(f"\nTotal text chunks created: {len(all_chunks)}")


# -------------------------------
# 🔎 Create embeddings & vector DB
# -------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

try:
    docsearch = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("✅ Documents ingested into Chroma successfully")

except Exception as e:
    print("❌ Error during vector ingestion:", e)
    traceback.print_exc()
    raise


# -------------------------------
# 🤖 Initialize LLM
# -------------------------------
llm = ChatOpenAI(
    model="gpt-4-1106-preview",
    openai_api_key=OPENAI_API_KEY,
    temperature=0
)


# -------------------------------
# 🔗 Create Conversational RAG Chain
# -------------------------------
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=docsearch.as_retriever()
)


# -------------------------------
# 💬 Interactive Chat Loop
# -------------------------------
chat_history = []

print("\n🤖 RAG Chatbot is ready! Type 'exit' to quit.\n")

while True:
    question = input("You: ")

    if question.lower() == "exit":
        break

    try:
        result = qa_chain.invoke({
            "question": question,
            "chat_history": chat_history
        })

        answer = result["answer"]
        print("\nAssistant:", answer, "\n")

        chat_history.append((question, answer))

    except Exception as e:
        print("❌ Error during QA:", e)
        traceback.print_exc()

print("👋 Exiting ChatBot")
