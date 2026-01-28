# api.py
import os, re, ast, sqlite3, traceback
from typing import List, Any, Optional
from io import BytesIO

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import pandas as pd

from dotenv import load_dotenv
load_dotenv()  # for OPENAI_API_KEY, etc.

# ---- LangChain / LLM bits ----
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI

# Optional RAG (PDF/TXT in DataFolder)
ENABLE_RAG = True
try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_text_splitters import CharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain.chains import ConversationalRetrievalChain
except Exception:
    ENABLE_RAG = False

# ---------- Config ----------
DATA_FOLDER = os.getenv("DATA_FOLDER", "DataFolder")
DB_PATH = os.getenv("SQLITE_DB_PATH", "sql_test.db")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Put it in your environment or a .env file."
    )

# ---------- FastAPI ----------
app = FastAPI(title="SQL Q API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev-friendly
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Excel -> SQLite helpers ----------
def load_excels_into_sqlite(db_path: str, data_folder: str):
    """
    Load every .xlsx/.xls in data_folder into SQLite; table name = filename (normalized).
    Requires `openpyxl` for .xlsx.
    """
    if not os.path.isdir(data_folder):
        print(f"[BOOT] Data folder not found: {data_folder} (skipping Excel load)")
        return

    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        for filename in os.listdir(data_folder):
            if filename.lower().endswith((".xlsx", ".xls")):
                file_path = os.path.join(data_folder, filename)
                try:
                    df = pd.read_excel(file_path)
                except ImportError as e:
                    raise RuntimeError(
                        "Missing Excel dependency. Install with: pip install openpyxl"
                    ) from e
                table_name = (
                    os.path.splitext(filename)[0]
                    .replace("-", "_")
                    .replace(" ", "_")
                    .lower()
                )
                df.to_sql(table_name, conn, if_exists="replace", index=False)
                print(f"[BOOT] Loaded Excel -> table '{table_name}' rows={len(df)}")
    finally:
        conn.close()

def run_sql(sql: str):
    # prevent silently creating an empty DB
    if not os.path.exists(DB_PATH):
        raise RuntimeError("SQLite DB not found. Ensure DataFolder has Excel files and restart the server.")
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        return cols, [list(r) for r in rows]
    finally:
        conn.close()

# ---------- Build SQL Agent (lazy) ----------
_sql_agent = None
def get_sql_agent():
    global _sql_agent
    if _sql_agent is None:
        if not os.path.exists(DB_PATH):
            raise RuntimeError("SQLite DB not found. Put Excel files in DataFolder and (re)start the server.")
        db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
        llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        _sql_agent = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=False,
            agent_executor_kwargs={
                "handle_parsing_errors": True,
                "return_intermediate_steps": False,
            },
            max_iterations=15
        )
    return _sql_agent

# ---------- Optional RAG Chat ----------
_rag_chain = None
def get_rag_chain():
    global _rag_chain
    if _rag_chain is not None:
        return _rag_chain
    if not ENABLE_RAG:
        return None
    try:
        if not os.path.isdir(DATA_FOLDER):
            print("[RAG] Data folder missing; starting without RAG.")
            return None
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = []
        for filename in os.listdir(DATA_FOLDER):
            path = os.path.join(DATA_FOLDER, filename)
            if filename.lower().endswith(".pdf"):
                try:
                    docs.extend(text_splitter.split_documents(PyPDFLoader(path).load()))
                except Exception:
                    traceback.print_exc()
            elif filename.lower().endswith(".txt"):
                try:
                    docs.extend(text_splitter.split_documents(TextLoader(path).load()))
                except Exception:
                    traceback.print_exc()
        if not docs:
            print("[RAG] No PDF/TXT docs found; starting without RAG.")
            return None
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vs = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
        llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
        _rag_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vs.as_retriever())
        print("[RAG] Vector store ready.")
        return _rag_chain
    except Exception:
        traceback.print_exc()
        return None

# ---------- x/y parser (pie chart) ----------
def safe_parse_xy_relaxed(output_text: str):
    x_match = re.search(r"x\s*=\s*(\[[^\]]*\])", output_text, flags=re.IGNORECASE|re.MULTILINE|re.DOTALL)
    y_match = re.search(r"y\s*=\s*(\[[^\]]*\])", output_text, flags=re.IGNORECASE|re.MULTILINE|re.DOTALL)
    if x_match and y_match:
        try:
            x_vals = ast.literal_eval(x_match.group(1))
            y_vals = ast.literal_eval(y_match.group(1))
            if isinstance(x_vals, list) and isinstance(y_vals, list) and len(x_vals) == len(y_vals):
                return x_vals, y_vals
        except Exception:
            pass
    pairs = re.findall(r"\('([^']+)'\s*,\s*([0-9]+)\)", output_text)
    if pairs:
        xs = [p[0] for p in pairs]
        ys = [int(p[1]) for p in pairs]
        if len(xs) == len(ys) and xs:
            return xs, ys
    return None, None

# ---------- Schemas ----------
class ChatReq(BaseModel):
    messages: List[dict]

class AskReq(BaseModel):
    question: str

class QueryReq(BaseModel):
    sql: str

class AskResp(BaseModel):
    answer: str
    sql: Optional[str] = None
    columns: List[str] = []
    rows: List[List[Any]] = []
    x: Optional[List[str]] = None
    y: Optional[List[float]] = None

# --- plotting models ---
import matplotlib
matplotlib.use("Agg")  # safe on servers/Windows without GUI
import matplotlib.pyplot as plt

class PlotStackedReq(BaseModel):
    # rows: list of [category, numeric]
    rows: List[List[Any]] = []
    # OR provide explicit bins or a step to auto-build bins
    bins: Optional[List[float]] = None
    step: Optional[int] = None
    # Optional labels/styling
    labels: Optional[List[str]] = None
    title: Optional[str] = "Stacked counts by numeric bins"
    xlabel: Optional[str] = "Range"
    ylabel: Optional[str] = "Count"
    colors: Optional[List[str]] = ["tomato", "skyblue"]

def _build_bins(step: Optional[int], default_max: float = 100000.0) -> Optional[List[float]]:
    if step and step > 0:
        return list(range(0, int(default_max) + step, step))
    return None

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"ok": True}

# NEW: list tables
@app.get("/tables")
def list_tables():
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=400, detail="DB not found. Add Excel files to DataFolder and restart.")
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        tables = [r[0] for r in cur.fetchall()]
        return {"tables": tables}
    finally:
        conn.close()

@app.post("/chat")
def chat(req: ChatReq):
    """
    If RAG is available, use it (PDF/TXT from DataFolder). Otherwise, plain LLM chat.
    """
    msgs = req.messages or []
    user_last = ""
    for m in reversed(msgs):
        if m.get("role") == "user":
            user_last = m.get("content", "")
            break

    rag = get_rag_chain()
    if rag and user_last:
        try:
            result = rag.invoke({"question": user_last, "chat_history": []})
            reply = result.get("answer", "(no answer)")
            return {"reply": reply}
        except Exception:
            traceback.print_exc()

    # Fallback: plain LLM
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    try:
        out = llm.invoke(user_last or "Hello")
        return {"reply": getattr(out, "content", str(out))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")

@app.post("/ask", response_model=AskResp)
def ask(req: AskReq):
    """
    Natural language -> (answer, sql, table, x/y)
    """
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question is empty.")

    agent = get_sql_agent()

    # 1) short textual answer
    text_prompt = (
        "Answer the user's analytics question briefly. "
        "If you must give a final answer, prefix with 'Final Answer:'. "
        "Question: " + q
    )
    try:
        text_raw = agent.invoke({"input": text_prompt})
        answer_text = text_raw.get("output", str(text_raw))
    except Exception:
        answer_text = "(No direct text answer)"

    # 2) get x/y lists for charts
    xy_prompt = (
        "You are a graph plotting assistant. You will be given a question and you "
        "need to generate EXACTLY two lists:\n"
        "x = [list]\n"
        "y = [list]\n"
        "Lengths must match. No extra text. SQLite SQL only. "
        "Question: " + q
    )
    x_vals = y_vals = None
    try:
        xy_raw = agent.invoke({"input": xy_prompt})
        raw = xy_raw.get("output", str(xy_raw))
        x_vals, y_vals = safe_parse_xy_relaxed(raw)
    except Exception:
        pass

    # 3) get a single best SQL
    sql_prompt = (
        "Write a single SQL (SQLite dialect) to answer:\n"
        f"{q}\n"
        "Return ONLY the SQL between triple backticks.\n```"
    )
    sql_text = None
    try:
        sql_raw = agent.invoke({"input": sql_prompt})
        body = sql_raw.get("output", str(sql_raw))
        m = re.search(r"```(.*?)```", body, flags=re.DOTALL)
        sql_text = (m.group(1) if m else body).strip().strip("`")
    except Exception:
        pass

    cols, rows = [], []
    if sql_text:
        try:
            cols, rows = run_sql(sql_text)
        except Exception:
            # Don’t crash if SQL is imperfect; still return x/y + textual answer
            pass

    return AskResp(
        answer=answer_text,
        sql=sql_text,
        columns=cols,
        rows=rows,
        x=x_vals,
        y=[float(v) for v in y_vals] if y_vals else None
    )

@app.post("/query")
def query(req: QueryReq):
    sql = (req.sql or "").strip()
    if not sql:
        raise HTTPException(status_code=400, detail="SQL is empty.")
    try:
        cols, rows = run_sql(sql)
        x = y = None
        lower = [c.lower() for c in cols]
        # expose x/y if caller uses label/value aliases
        if "label" in lower and "value" in lower:
            li = lower.index("label")
            vi = lower.index("value")
            x = [str(r[li]) for r in rows]
            y = [float(r[vi]) if r[vi] is not None else 0 for r in rows]
        return {"columns": cols, "rows": rows, "x": x, "y": y}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"SQL error: {e}")

# ---------- Server-side generic stacked plot ----------
@app.post("/plot/stacked")
def plot_stacked(req: PlotStackedReq):
    """
    Server-rendered PNG stacked bar chart.
    Input:
      - rows: list of [category, numeric]
      - bins OR step (to auto-build bins); if neither, auto 10 bins over observed range
    Returns:
      - PNG image (image/png)
    """
    if not req.rows or not isinstance(req.rows, list):
        raise HTTPException(status_code=400, detail="rows must be a non-empty list like [['female', 24.3], ['male', 30.1], ...].")

    # Build DataFrame
    try:
        df = pd.DataFrame(req.rows, columns=["category", "value"])
        df["category"] = df["category"].astype(str)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse rows: {e}")

    # Bins
    bins = req.bins or _build_bins(req.step)
    if bins is None:
        # auto bins across observed range
        vmin, vmax = float(df["value"].min()), float(df["value"].max())
        if vmin == vmax:
            vmax = vmin + 1.0
        import numpy as np
        bins = list(np.linspace(vmin, vmax, 11))
    if len(bins) < 2:
        raise HTTPException(status_code=400, detail="Provide at least two bin edges in 'bins' or a positive 'step'.")

    # Labels
    labels = req.labels
    if labels is None:
        labels = [f"{bins[i]:.2f}–{bins[i+1]:.2f}" for i in range(len(bins) - 1)]
    if len(labels) != len(bins) - 1:
        raise HTTPException(status_code=400, detail="labels length must be len(bins) - 1.")

    # Cut and group
    df["bin"] = pd.cut(df["value"], bins=bins, labels=labels, include_lowest=True)
    grouped = df.groupby(["bin", "category"]).size().unstack(fill_value=0).sort_index()

    cats = list(grouped.columns)
    colors = (req.colors or ["tomato", "skyblue"]).copy()
    if len(colors) < len(cats):
        times = (len(cats) + len(colors) - 1) // len(colors)
        colors = (colors * times)[:len(cats)]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = None
    for i, cat in enumerate(cats):
        vals = grouped[cat].values
        ax.bar(grouped.index.astype(str), vals, bottom=bottom, label=cat, color=colors[i])
        bottom = vals if bottom is None else (bottom + vals)

    ax.set_title(req.title or "Stacked counts by numeric bins", fontsize=14)
    ax.set_xlabel(req.xlabel or "Range")
    ax.set_ylabel(req.ylabel or "Count")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

# ---------- App startup ----------
@app.on_event("startup")
def _startup():
    print("[BOOT] Loading Excel -> SQLite …")
    load_excels_into_sqlite(DB_PATH, DATA_FOLDER)
    if ENABLE_RAG:
        _ = get_rag_chain()
    print("[BOOT] Ready.")
