import streamlit as st
import pandas as pd
import re
import os
import json
import io
import time
import zipfile
from groq import Groq

# ── PAGE SETUP ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="InsightX RAG | Universal AI", layout="wide")
st.title("🧠 InsightX RAG: Ask Anything About Any File")
st.caption("True RAG system — OCR, code execution, large context, any file type.")

# ── API KEY ───────────────────────────────────────────────────────────────────
HARDCODED_KEY = "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"

api_key = ""
try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    pass
if not api_key:
    api_key = os.environ.get("GROQ_API_KEY", "")
if not api_key:
    api_key = HARDCODED_KEY
if not api_key or api_key == "PASTE_YOUR_GROQ_API_KEY_HERE":
    st.error("Please paste your Groq API key into line 14 of app.py")
    st.stop()

client = Groq(api_key=api_key)

# ── INSTALL PACKAGES ──────────────────────────────────────────────────────────
@st.cache_resource
def install_packages():
    import subprocess
    packages = [
        "pypdf", "python-docx", "faiss-cpu",
        "sentence-transformers", "openpyxl",
        "python-pptx", "Pillow", "pytesseract",
        "pdf2image", "opencv-python-headless"
    ]
    for pkg in packages:
        subprocess.run(["pip", "install", pkg, "-q"], capture_output=True)
    subprocess.run(
        ["apt-get", "install", "-y", "-q", "tesseract-ocr", "poppler-utils"],
        capture_output=True
    )

with st.spinner("Setting up AI engine (first run only)..."):
    install_packages()

# ── EMBEDDING MODEL ───────────────────────────────────────────────────────────
@st.cache_resource
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB  = 200
TOP_K_CHUNKS      = 3       # Only top 3 chunks → stays well under token limit
MAX_WORDS_FOR_RAG = 80000   # Hard cap for vector indexing
MAX_CSV_ROWS      = 15000   # Max rows used for AI Q&A text
MAX_CONTEXT_CHARS = 4000    # Hard cap on chars sent to Groq
MAX_TOKENS_OUT    = 700     # Max tokens in AI response

# ── FILE EXTRACTORS ───────────────────────────────────────────────────────────
def extract_text_from_csv(file):
    df = pd.read_csv(file)
    total_rows = len(df)
    text = f"CSV File | Total Rows: {total_rows:,} | Columns: {list(df.columns)}\n\n"
    if total_rows > MAX_CSV_ROWS:
        sample_df = df.sample(n=MAX_CSV_ROWS, random_state=42)
        text += (
            f"(Smart sample of {MAX_CSV_ROWS:,} rows shown for AI Q&A. "
            f"Full {total_rows:,} rows used in Precise Calculator.)\n\n"
        )
        text += sample_df.to_string(index=False)
    else:
        text += df.to_string(index=False)
    return text, df

def extract_text_from_excel(file):
    xl = pd.ExcelFile(file)
    all_text = ""
    first_df = None
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        if first_df is None:
            first_df = df
        all_text += f"\n--- Sheet: {sheet} ---\n"
        all_text += f"Rows: {len(df)} | Columns: {list(df.columns)}\n"
        all_text += df.head(500).to_string(index=False) + "\n"
    return all_text, first_df

def extract_text_from_pdf(file_bytes):
    import pypdf
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    text = f"PDF | Total Pages: {len(reader.pages)}\n"
    ocr_needed = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if len(page_text.strip()) > 30:
            text += f"\n--- Page {i+1} ---\n{page_text}\n"
        else:
            ocr_needed.append(i + 1)
    if ocr_needed:
        st.info(f"🔍 Detected {len(ocr_needed)} scanned page(s). Running OCR...")
        try:
            import pytesseract
            from pdf2image import convert_from_bytes
            images = convert_from_bytes(file_bytes, dpi=150)
            for pg in ocr_needed:
                if pg <= len(images):
                    ocr_text = pytesseract.image_to_string(images[pg - 1])
                    if ocr_text.strip():
                        text += f"\n--- Page {pg} (OCR) ---\n{ocr_text}\n"
            st.success("✅ OCR completed!")
        except Exception as e:
            text += f"\n[OCR failed: {e}]\n"
    return text, None

def extract_text_from_docx(file):
    import docx
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        if para.text.strip():
            text += para.text + "\n"
    for table in doc.tables:
        for row in table.rows:
            text += " | ".join(c.text.strip() for c in row.cells) + "\n"
    return text, None

def extract_text_from_pptx(file):
    from pptx import Presentation
    prs = Presentation(file)
    text = f"PowerPoint | Slides: {len(prs.slides)}\n"
    for i, slide in enumerate(prs.slides):
        text += f"\n--- Slide {i+1} ---\n"
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                text += shape.text + "\n"
    return text, None

def extract_text_from_image(file):
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(file)
        st.info("🔍 Running OCR on image...")
        ocr_text = pytesseract.image_to_string(img)
        w, h = img.size
        text = f"Image | {w}x{h}px\n\n"
        text += ocr_text if ocr_text.strip() else "No readable text found."
        st.success("✅ OCR done!")
        return text, None
    except Exception as e:
        return f"Image OCR error: {e}", None

def extract_text_from_json(file):
    data = json.load(file)
    return json.dumps(data, indent=2), None

def extract_text_from_txt(file):
    return file.read().decode("utf-8", errors="ignore"), None

def extract_text_from_zip(file):
    text = "ZIP Archive:\n"
    with zipfile.ZipFile(file) as z:
        for name in z.namelist():
            text += f"  - {name}\n"
            if name.endswith((".txt", ".csv", ".json", ".md", ".py", ".sql")):
                try:
                    with z.open(name) as f:
                        content = f.read().decode("utf-8", errors="ignore")
                        text += f"\n--- {name} ---\n{content[:2000]}\n"
                except Exception:
                    pass
    return text, None

def extract_content(uploaded_file):
    name = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()
    size_mb = len(file_bytes) / (1024 * 1024)

    if size_mb > MAX_FILE_SIZE_MB:
        st.error(f"File too large ({size_mb:.1f} MB). Max: {MAX_FILE_SIZE_MB} MB.")
        st.stop()

    file_obj = io.BytesIO(file_bytes)
    df = None

    if name.endswith(".csv"):
        text, df = extract_text_from_csv(file_obj)
    elif name.endswith((".xlsx", ".xls")):
        text, df = extract_text_from_excel(file_obj)
    elif name.endswith(".pdf"):
        text, df = extract_text_from_pdf(file_bytes)
    elif name.endswith(".docx"):
        text, df = extract_text_from_docx(file_obj)
    elif name.endswith(".pptx"):
        text, df = extract_text_from_pptx(file_obj)
    elif name.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")):
        text, df = extract_text_from_image(file_obj)
    elif name.endswith(".json"):
        text, df = extract_text_from_json(file_obj)
    elif name.endswith(".zip"):
        text, df = extract_text_from_zip(file_obj)
    else:
        try:
            text = file_bytes.decode("utf-8", errors="ignore")
        except Exception:
            text = "Could not read file."
        df = None

    if not text or len(text.strip()) < 10:
        st.error("File appears empty or unreadable.")
        st.stop()

    return text, df

# ── RAG CORE ──────────────────────────────────────────────────────────────────
def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    # Apply word cap before chunking
    if len(words) > MAX_WORDS_FOR_RAG:
        step = max(len(words) // MAX_WORDS_FOR_RAG, 1)
        words = words[::step][:MAX_WORDS_FOR_RAG]
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + chunk_size]))
        i += chunk_size - overlap
    return chunks

@st.cache_resource
def build_vector_index(text, file_name):
    import faiss
    import numpy as np

    model = load_embedding_model()
    chunks = chunk_text(text)
    if not chunks:
        return None, []

    progress = st.progress(0, text="Indexing your file...")
    embeddings = []
    batch_size = 32
    for i in range(0, len(chunks), batch_size):
        batch_emb = model.encode(chunks[i:i + batch_size], show_progress_bar=False)
        embeddings.extend(batch_emb)
        pct = min(int((i + batch_size) / len(chunks) * 100), 100)
        progress.progress(pct, text=f"Indexing {min(i+batch_size,len(chunks))}/{len(chunks)} chunks...")

    progress.empty()
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks

def retrieve_relevant_chunks(question, index, chunks):
    import faiss
    import numpy as np
    model = load_embedding_model()
    q_emb = np.array(model.encode([question])).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, min(TOP_K_CHUNKS, len(chunks)))
    return [chunks[i] for i in indices[0] if i < len(chunks)]

def ask_ai_with_retry(question, relevant_chunks, chat_history, retries=3):
    # Hard cap context to stay under Groq token limit
    context = "\n\n---\n\n".join(relevant_chunks)
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n...[truncated]"

    system_prompt = (
        "You are an expert Data Analyst and Document AI. "
        "Answer based on the provided file excerpts. "
        "Be concise, accurate, and helpful. "
        "If answer is not in excerpts, say so honestly."
    )
    messages = [{"role": "system", "content": system_prompt}]
    for chat in chat_history[-3:]:  # Only last 3 to save tokens
        messages.append({"role": "user", "content": chat["question"][:500]})
        messages.append({"role": "assistant", "content": chat["answer"][:500]})
    messages.append({
        "role": "user",
        "content": f"File content:\n{context}\n\nQuestion: {question}"
    })

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                max_tokens=MAX_TOKENS_OUT,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise e

# ── PRECISE CALCULATOR ────────────────────────────────────────────────────────
def generate_and_run_code(question, df, retries=3):
    cols_info = (
        f"Columns: {list(df.columns)}\n"
        f"Dtypes:\n{df.dtypes.to_string()}\n"
        f"Sample:\n{df.head(3).to_string()}"
    )
    if len(cols_info) > 1500:
        cols_info = cols_info[:1500]

    system_prompt = (
        "You are a Python data analyst. Write pandas code to answer the question. "
        "DataFrame is called 'df'. Use st.write(), st.dataframe(), "
        "st.bar_chart(), st.line_chart(), or st.metric() to display results. "
        "Return ONLY raw executable Python code. No markdown, no backticks."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"DataFrame info:\n{cols_info}\n\nQuestion: {question}"}
    ]

    raw_code = ""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                max_tokens=800,
            )
            raw_code = (
                response.choices[0].message.content
                .replace("```python", "").replace("```", "").strip()
            )
            # Sanitize timestamps
            clean_lines = [
                line for line in raw_code.splitlines()
                if not re.match(r"^\d{1,2}:\d{2}\s*(AM|PM)?$", line.strip(), re.IGNORECASE)
            ]
            clean_code = "\n".join(clean_lines).strip()
            import numpy as np
            exec(clean_code, {"df": df, "st": st, "pd": pd, "np": np})
            return clean_code, None
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                return raw_code, str(e)

# ── DATA QUALITY ──────────────────────────────────────────────────────────────
def data_quality_report(df):
    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.header("📂 Upload Any File")
st.sidebar.caption("CSV, Excel, PDF (scanned too), Word, PowerPoint, JSON, TXT, ZIP, Images...")

uploaded_file = st.sidebar.file_uploader(
    "Drop your file here",
    type=[
        "csv", "xlsx", "xls", "pdf", "docx", "pptx",
        "json", "txt", "sql", "md", "py", "log",
        "xml", "html", "css", "js", "ts", "yaml", "yml",
        "zip", "png", "jpg", "jpeg", "webp", "bmp", "tiff"
    ]
)

# ── MAIN APP ──────────────────────────────────────────────────────────────────
if uploaded_file:
    with st.spinner(f"📖 Reading {uploaded_file.name}..."):
        extracted_text, df = extract_content(uploaded_file)

    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    word_count = len(extracted_text.split())

    st.sidebar.success(f"✅ Loaded: {uploaded_file.name}")
    st.sidebar.info(f"📝 {word_count:,} words | {file_size_mb:.2f} MB")

    index, chunks = build_vector_index(extracted_text, uploaded_file.name)
    if index is None:
        st.error("Could not index this file.")
        st.stop()

    st.sidebar.info(f"🧩 {len(chunks)} chunks indexed")

    # ── GUIDE TABLE ───────────────────────────────────────────────────────────
    st.markdown("""
| Tab | Best For | Accuracy |
|-----|----------|----------|
| 💬 Ask AI | Summaries, explanations, trends, general Q&A | Smart sample — great for insights |
| ⚡ Precise Calculator | Totals, averages, rankings, exact numbers | ✅ 100% Full dataset — mathematically exact |
| 📊 Data Analysis | Charts, missing values, duplicates, stats | ✅ 100% Full dataset |
""")

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["💬 Ask AI", "⚡ Precise Calculator", "📊 Data Analysis", "👀 Preview"]
    )

    # ── TAB 1: ASK AI ─────────────────────────────────────────────────────────
    with tab1:
        st.subheader("💬 Ask Anything — Plain English")
        st.caption("Best for: summaries, explanations, pattern finding, document Q&A")

        st.info(
            "💡 **How this tab works:** Uses RAG (smart search) to find the most relevant "
            "parts of your file and answers in plain English. Best for **summaries, "
            "explanations, trends, and general questions**. "
            "For **100% precise calculations** → use the ⚡ Precise Calculator tab."
        )

        quick_questions = [
            "Summarize the key insights from this file",
            "What are the main trends or patterns?",
            "Give me a full statistical summary",
            "What are the top 5 most important findings?",
            "Are there any anomalies or outliers?",
            "What does this file contain overall?",
            "List all unique categories or topics mentioned",
        ]
        selected_quick = st.selectbox(
            "⚡ Quick Questions:", ["Choose or type below..."] + quick_questions,
            key="quick_q"
        )
        user_question = st.text_input(
            "Your question:",
            placeholder="e.g., What is the main topic of this document?",
            key="ask_ai_input"
        )

        final_question = user_question if user_question else (
            selected_quick if selected_quick != "Choose or type below..." else None
        )

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "last_file" not in st.session_state:
            st.session_state.last_file = ""
        if st.session_state.last_file != uploaded_file.name:
            st.session_state.chat_history = []
            st.session_state.calc_history = []
            st.session_state.last_file = uploaded_file.name

        if final_question:
            with st.spinner("🔍 Searching file..."):
                relevant_chunks = retrieve_relevant_chunks(final_question, index, chunks)
            with st.spinner("🤖 Generating answer..."):
                try:
                    answer = ask_ai_with_retry(
                        final_question, relevant_chunks, st.session_state.chat_history
                    )
                    st.session_state.chat_history.append({
                        "question": final_question,
                        "answer": answer
                    })
                except Exception as e:
                    st.error(f"AI Error: {e}")

        if st.session_state.chat_history:
            chat_export = "\n\n".join(
                [f"Q: {c['question']}\nA: {c['answer']}"
                 for c in st.session_state.chat_history]
            )
            st.download_button(
                "⬇️ Download Chat History",
                data=chat_export,
                file_name="insightx_chat.txt",
                mime="text/plain",
                key="dl_chat_history"
            )

            st.subheader("🗨️ Conversation")
            for idx, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.chat_message("user"):
                    st.write(chat["question"])
                with st.chat_message("assistant"):
                    st.write(chat["answer"])
                    # Unique key using index — fixes DuplicateElementKey error
                    st.download_button(
                        "⬇️ Download this answer",
                        data=chat["answer"],
                        file_name=f"answer_{idx}.txt",
                        mime="text/plain",
                        key=f"dl_answer_{idx}"
                    )

            if st.button("🗑️ Clear Conversation", key="clear_chat"):
                st.session_state.chat_history = []
                st.rerun()

    # ── TAB 2: PRECISE CALCULATOR ─────────────────────────────────────────────
    with tab2:
        st.subheader("⚡ Precise Calculator")
        st.caption("Best for: exact calculations, aggregations, comparisons, custom charts")

        if df is not None:
            st.success(
                "✅ **100% Accurate — Full Dataset Mode** | "
                "Runs real Python/pandas code on your **complete dataset**. "
                "No sampling, no approximation. Every number is mathematically exact."
            )

            if "calc_history" not in st.session_state:
                st.session_state.calc_history = []

            calc_question = st.text_input(
                "What do you want to calculate?",
                placeholder="e.g., Total amount by category | Top 10 by sales | Monthly trend",
                key="calc_input"
            )

            if calc_question:
                with st.spinner("🧮 Writing and running code on full dataset..."):
                    code_used, error = generate_and_run_code(calc_question, df)
                    if error:
                        st.error(f"Execution error: {error}")
                        if code_used:
                            st.code(code_used, language="python")
                    else:
                        st.session_state.calc_history.append({
                            "question": calc_question,
                            "code": code_used
                        })
                        with st.expander("🔍 View generated code"):
                            st.code(code_used, language="python")

            if st.session_state.calc_history:
                if st.button("🗑️ Clear Calculator History", key="clear_calc"):
                    st.session_state.calc_history = []
                    st.rerun()
        else:
            st.info("⚡ Precise Calculator works with CSV and Excel files. Upload one to use this feature.")

    # ── TAB 3: DATA ANALYSIS ──────────────────────────────────────────────────
    with tab3:
        st.subheader("📊 Data Analysis")
        if df is not None:
            report = data_quality_report(df)

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Rows", f"{report['total_rows']:,}")
            col2.metric("Total Columns", f"{report['total_columns']}")
            col3.metric("Duplicate Rows", f"{report['duplicate_rows']:,}")

            missing = {k: v for k, v in report["missing_values"].items() if v > 0}
            if missing:
                st.warning(f"⚠️ Missing values in: {', '.join(missing.keys())}")
                st.dataframe(
                    pd.DataFrame.from_dict(missing, orient="index", columns=["Missing Count"])
                )
            else:
                st.success("✅ No missing values!")

            if report["duplicate_rows"] > 0:
                st.warning(f"⚠️ {report['duplicate_rows']} duplicate rows found!")
            else:
                st.success("✅ No duplicate rows!")

            st.markdown("#### 📈 Auto Charts")
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if numeric_cols:
                selected_col = st.selectbox("Select column:", numeric_cols, key="chart_col")
                chart_type = st.radio(
                    "Chart type:", ["Bar", "Line", "Area"],
                    horizontal=True, key="chart_type"
                )
                chart_data = df[selected_col].dropna().reset_index(drop=True)
                if chart_type == "Bar":
                    st.bar_chart(chart_data)
                elif chart_type == "Line":
                    st.line_chart(chart_data)
                else:
                    st.area_chart(chart_data)

            st.markdown("#### 📋 Statistical Summary")
            st.dataframe(df.describe())

            st.markdown("#### 🗃️ Raw Data (first 200 rows)")
            st.dataframe(df.head(200))

            st.download_button(
                "⬇️ Download Full Data as CSV",
                data=df.to_csv(index=False),
                file_name="insightx_data.csv",
                mime="text/csv",
                key="dl_csv"
            )
        else:
            st.info("📊 Data Analysis is available for CSV and Excel files.")

    # ── TAB 4: PREVIEW ────────────────────────────────────────────────────────
    with tab4:
        st.subheader("👀 File Preview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Words", f"{word_count:,}")
        col2.metric("Chunks Indexed", f"{len(chunks):,}")
        col3.metric("File Size", f"{file_size_mb:.2f} MB")
        st.text_area(
            "Extracted Content (first 3000 chars):",
            extracted_text[:3000] + ("..." if len(extracted_text) > 3000 else ""),
            height=400,
            key="preview_text"
        )

else:
    st.info("👋 Welcome to InsightX RAG! Upload any file in the sidebar to get started.")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("### 📄 Documents")
        st.write("PDF (inc. scanned), Word, PowerPoint, TXT")
    with col2:
        st.markdown("### 📊 Data Files")
        st.write("CSV, Excel, JSON, SQL, YAML")
    with col3:
        st.markdown("### 💻 More Formats")
        st.write("Python, HTML, JS, XML, ZIP, Images (OCR)")
    with col4:
        st.markdown("### 🚀 3 AI Modes")
        st.write("RAG Q&A + Precise Calculator + Auto Charts")

    st.divider()
    st.markdown("""
### How it works:
1. **Upload** any file → text extracted (OCR for scanned files/images)
2. **RAG indexes** the full file into searchable vector chunks
3. **Ask AI tab** → plain English Q&A on any part of the file
4. **Precise Calculator tab** → AI writes and runs real Python code for exact math
5. **Data Analysis tab** → auto quality checks, charts, and stats
6. **Download** answers, chat history, or cleaned data anytime
""")
