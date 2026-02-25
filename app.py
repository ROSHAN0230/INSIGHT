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
HARDCODED_KEY = "AIzaSyDBLWRjQPhdwKWIPU39eJa9921jJAFg7m4"

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
    # Install tesseract OCR engine
    subprocess.run(["apt-get", "install", "-y", "-q", "tesseract-ocr", "poppler-utils"],
                   capture_output=True)

with st.spinner("Setting up AI engine (first run only)..."):
    install_packages()

# ── LAZY IMPORTS ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = 200
TOP_K_CHUNKS = 8  # Increased from 5 for larger context

# ── FILE EXTRACTORS ───────────────────────────────────────────────────────────
def extract_text_from_csv(file):
    df = pd.read_csv(file)
    text = f"CSV File | Rows: {len(df)} | Columns: {list(df.columns)}\n\n"
    text += df.to_string(index=False)
    return text, df

def extract_text_from_excel(file):
    xl = pd.ExcelFile(file)
    all_text = ""
    dfs = {}
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        dfs[sheet] = df
        all_text += f"\n--- Sheet: {sheet} ---\n"
        all_text += f"Rows: {len(df)} | Columns: {list(df.columns)}\n"
        all_text += df.to_string(index=False) + "\n"
    first_df = list(dfs.values())[0] if dfs else None
    return all_text, first_df

def extract_text_from_pdf(file):
    """Extract text from PDF — tries normal extraction first, falls back to OCR."""
    import pypdf
    file_bytes = file.read() if hasattr(file, 'read') else file
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    text = f"PDF | Total Pages: {len(reader.pages)}\n"
    ocr_needed_pages = []

    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if len(page_text.strip()) > 30:
            text += f"\n--- Page {i+1} ---\n{page_text}\n"
        else:
            ocr_needed_pages.append(i + 1)

    # OCR fallback for image-based/scanned pages
    if ocr_needed_pages:
        st.info(f"🔍 Detected {len(ocr_needed_pages)} scanned page(s). Running OCR...")
        try:
            import pytesseract
            from pdf2image import convert_from_bytes
            images = convert_from_bytes(file_bytes, dpi=200)
            for page_num in ocr_needed_pages:
                if page_num <= len(images):
                    img = images[page_num - 1]
                    ocr_text = pytesseract.image_to_string(img)
                    if ocr_text.strip():
                        text += f"\n--- Page {page_num} (OCR) ---\n{ocr_text}\n"
            st.success("✅ OCR completed!")
        except Exception as e:
            text += f"\n[OCR failed for pages {ocr_needed_pages}: {e}]\n"

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
            text += " | ".join(cell.text.strip() for cell in row.cells) + "\n"
    return text, None

def extract_text_from_pptx(file):
    from pptx import Presentation
    prs = Presentation(file)
    text = f"PowerPoint | Total Slides: {len(prs.slides)}\n"
    for i, slide in enumerate(prs.slides):
        text += f"\n--- Slide {i+1} ---\n"
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                text += shape.text + "\n"
    return text, None

def extract_text_from_image(file):
    """Extract text from images using OCR."""
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(file)
        st.info("🔍 Running OCR on image...")
        ocr_text = pytesseract.image_to_string(img)
        width, height = img.size
        text = f"Image File | Size: {width}x{height}px\n\n"
        if ocr_text.strip():
            text += f"--- OCR Extracted Text ---\n{ocr_text}\n"
            st.success("✅ OCR completed!")
        else:
            text += "No readable text found in image.\n"
        return text, None
    except Exception as e:
        return f"Image OCR error: {e}", None

def extract_text_from_json(file):
    data = json.load(file)
    return json.dumps(data, indent=2), None

def extract_text_from_txt(file):
    return file.read().decode("utf-8", errors="ignore"), None

def extract_text_from_zip(file):
    text = "ZIP Archive Contents:\n"
    with zipfile.ZipFile(file) as z:
        for name in z.namelist():
            text += f"  - {name}\n"
            if name.endswith((".txt", ".csv", ".json", ".md", ".py", ".sql", ".html", ".xml")):
                try:
                    with z.open(name) as f:
                        content = f.read().decode("utf-8", errors="ignore")
                        text += f"\n--- {name} ---\n{content[:3000]}\n"
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
    elif name.endswith((".txt", ".md", ".sql", ".py", ".log", ".xml",
                        ".html", ".css", ".js", ".ts", ".yaml", ".yml")):
        text, df = extract_text_from_txt(file_obj)
    else:
        try:
            text = file_bytes.decode("utf-8", errors="ignore")
        except Exception:
            text = "Could not read this file type."
        df = None

    if not text or len(text.strip()) < 10:
        st.error("The file appears to be empty or unreadable.")
        st.stop()

    return text, df

# ── RAG CORE ──────────────────────────────────────────────────────────────────
def chunk_text(text, chunk_size=500, overlap=75):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
        if i >= len(words):
            break
    return chunks

@st.cache_resource
def build_vector_index(text, file_name, chunk_size=500):
    import faiss
    import numpy as np

    model = load_embedding_model()
    chunks = chunk_text(text, chunk_size=chunk_size)
    if not chunks:
        return None, []

    progress = st.progress(0, text="Indexing your file...")
    embeddings = []
    batch_size = 32
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_emb = model.encode(batch, show_progress_bar=False)
        embeddings.extend(batch_emb)
        pct = min(int((i + batch_size) / len(chunks) * 100), 100)
        progress.progress(pct, text=f"Indexing {min(i+batch_size, len(chunks))}/{len(chunks)} chunks...")

    progress.empty()
    import numpy as np
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks

def retrieve_relevant_chunks(question, index, chunks, top_k=TOP_K_CHUNKS):
    import faiss
    import numpy as np

    model = load_embedding_model()
    q_emb = model.encode([question])
    q_emb = np.array(q_emb).astype("float32")
    faiss.normalize_L2(q_emb)

    scores, indices = index.search(q_emb, min(top_k, len(chunks)))
    return [chunks[i] for i in indices[0] if i < len(chunks)]

def ask_ai_with_retry(question, relevant_chunks, chat_history, retries=3):
    context = "\n\n---\n\n".join(relevant_chunks)

    system_prompt = (
        "You are an expert Data Analyst and Document AI. "
        "You are given relevant excerpts from a large document retrieved based on the user's question. "
        "Answer the user's question accurately and thoroughly. "
        "If calculating statistics, be precise. "
        "If summarizing, be structured and clear. "
        "If the answer is not in the excerpts, say so honestly. "
        "Be helpful, concise, and accurate."
    )

    messages = [{"role": "system", "content": system_prompt}]
    for chat in chat_history[-6:]:
        messages.append({"role": "user", "content": chat["question"]})
        messages.append({"role": "assistant", "content": chat["answer"]})
    messages.append({
        "role": "user",
        "content": f"Relevant file content:\n{context}\n\nQuestion: {question}"
    })

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                max_tokens=2048,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise e

# ── PYTHON CODE EXECUTOR ──────────────────────────────────────────────────────
def generate_and_run_code(question, df, chat_history, retries=3):
    """Ask AI to generate Python/pandas code and execute it precisely."""
    columns_info = f"Columns: {list(df.columns)}\nDtypes:\n{df.dtypes.to_string()}\nSample:\n{df.head(3).to_string()}"

    system_prompt = (
        "You are a Python data analyst. Given a pandas DataFrame called 'df', "
        "write Python code to answer the user's question precisely. "
        "Use pandas and numpy for calculations. "
        "Use st.write(), st.dataframe(), st.bar_chart(), st.line_chart(), or st.metric() to display results. "
        "Return ONLY raw executable Python code. No markdown, no backticks, no explanation."
    )

    messages = [{"role": "system", "content": system_prompt}]
    for chat in chat_history[-4:]:
        messages.append({"role": "user", "content": chat["question"]})
        messages.append({"role": "assistant", "content": chat["answer"]})
    messages.append({
        "role": "user",
        "content": f"DataFrame info:\n{columns_info}\n\nQuestion: {question}"
    })

    raw_code = ""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                max_tokens=1024,
            )
            raw_code = (
                response.choices[0].message.content
                .replace("```python", "").replace("```", "").strip()
            )
            # Sanitize timestamps
            clean_lines = []
            for line in raw_code.splitlines():
                if re.match(r"^\d{1,2}:\d{2}\s*(AM|PM)?$", line.strip(), re.IGNORECASE):
                    continue
                clean_lines.append(line)
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
        "data_types": df.dtypes.astype(str).to_dict(),
    }

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.header("📂 Upload Any File")
st.sidebar.caption("CSV, Excel, PDF (inc. scanned), Word, PowerPoint, JSON, TXT, SQL, ZIP, Images, Code...")

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

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Ask AI", "⚡ Precise Calculator", "📊 Data Analysis", "👀 Preview"])

    # ── TAB 1: ASK AI (RAG) ───────────────────────────────────────────────────
    with tab1:
        st.subheader("💬 Ask Anything — Plain English")
        st.caption("Best for: summaries, explanations, pattern finding, document Q&A")

        quick_questions = [
            "Summarize the key insights from this file",
            "What are the main trends or patterns?",
            "Give me a full statistical summary",
            "What are the top 5 most important findings?",
            "Are there any anomalies or outliers?",
            "What does this file contain overall?",
            "List all unique categories or topics mentioned",
        ]
        selected_quick = st.selectbox("⚡ Quick Questions:", ["Choose or type below..."] + quick_questions)
        user_question = st.text_input("Your question:", placeholder="e.g., What is the main topic of this document?")

        final_question = user_question if user_question else (
            selected_quick if selected_quick != "Choose or type below..." else None
        )

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "last_file" not in st.session_state:
            st.session_state.last_file = ""
        if st.session_state.last_file != uploaded_file.name:
            st.session_state.chat_history = []
            st.session_state.last_file = uploaded_file.name

        if final_question:
            with st.spinner("🔍 Retrieving relevant content..."):
                relevant_chunks = retrieve_relevant_chunks(final_question, index, chunks)
            with st.spinner("🤖 Generating answer..."):
                try:
                    answer = ask_ai_with_retry(final_question, relevant_chunks, st.session_state.chat_history)
                    st.session_state.chat_history.append({"question": final_question, "answer": answer})
                except Exception as e:
                    st.error(f"AI Error: {e}")

        if st.session_state.chat_history:
            chat_export = "\n\n".join([f"Q: {c['question']}\nA: {c['answer']}" for c in st.session_state.chat_history])
            st.download_button("⬇️ Download Chat History", data=chat_export,
                               file_name="insightx_chat.txt", mime="text/plain")

            st.subheader("🗨️ Conversation")
            for chat in reversed(st.session_state.chat_history):
                with st.chat_message("user"):
                    st.write(chat["question"])
                with st.chat_message("assistant"):
                    st.write(chat["answer"])
                    st.download_button("⬇️ Download answer", data=chat["answer"],
                                       file_name="answer.txt", mime="text/plain",
                                       key=f"dl_{hash(chat['question'])}")
            if st.button("🗑️ Clear Conversation"):
                st.session_state.chat_history = []
                st.rerun()

    # ── TAB 2: PRECISE CALCULATOR (CODE EXECUTOR) ─────────────────────────────
    with tab2:
        st.subheader("⚡ Precise Calculator")
        st.caption("Best for: exact calculations, aggregations, comparisons, custom charts on CSV/Excel data")

        if df is not None:
            st.info("💡 Ask precise math questions — the AI writes and runs actual Python/pandas code on your data for 100% accurate results.")

            calc_question = st.text_input(
                "What do you want to calculate?",
                placeholder="e.g., Average salary by department | Total revenue per month | Top 10 by sales"
            )

            if "calc_history" not in st.session_state:
                st.session_state.calc_history = []

            if calc_question:
                with st.spinner("🧮 Generating and running code..."):
                    code_used, error = generate_and_run_code(
                        calc_question, df, st.session_state.calc_history
                    )
                    if error:
                        st.error(f"Execution error: {error}")
                        st.code(code_used, language="python")
                    else:
                        st.session_state.calc_history.append({
                            "question": calc_question,
                            "answer": f"[Code executed successfully]\n```python\n{code_used}\n```"
                        })
                        with st.expander("🔍 View generated code"):
                            st.code(code_used, language="python")

            if st.session_state.calc_history:
                if st.button("🗑️ Clear Calculator History"):
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
                st.dataframe(pd.DataFrame.from_dict(missing, orient="index", columns=["Missing Count"]))
            else:
                st.success("✅ No missing values!")

            if report["duplicate_rows"] > 0:
                st.warning(f"⚠️ {report['duplicate_rows']} duplicate rows found!")
            else:
                st.success("✅ No duplicate rows!")

            st.markdown("#### 📈 Auto Charts")
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if numeric_cols:
                selected_col = st.selectbox("Select column:", numeric_cols)
                chart_type = st.radio("Chart type:", ["Bar", "Line", "Area"], horizontal=True)
                chart_data = df[selected_col].dropna().reset_index(drop=True)
                if chart_type == "Bar":
                    st.bar_chart(chart_data)
                elif chart_type == "Line":
                    st.line_chart(chart_data)
                else:
                    st.area_chart(chart_data)

            st.markdown("#### 📋 Statistical Summary")
            st.dataframe(df.describe())

            st.markdown("#### 🗃️ Raw Data")
            st.dataframe(df.head(200))

            csv_out = df.to_csv(index=False)
            st.download_button("⬇️ Download as CSV", data=csv_out,
                               file_name="insightx_data.csv", mime="text/csv")
        else:
            st.info("📊 Data Analysis is available for CSV and Excel files.")

    # ── TAB 4: PREVIEW ────────────────────────────────────────────────────────
    with tab4:
        st.subheader("👀 File Preview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Words", f"{word_count:,}")
        col2.metric("Chunks", f"{len(chunks):,}")
        col3.metric("Size", f"{file_size_mb:.2f} MB")
        st.text_area("Extracted Content (first 3000 chars):",
                     extracted_text[:3000] + ("..." if len(extracted_text) > 3000 else ""),
                     height=400)

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
