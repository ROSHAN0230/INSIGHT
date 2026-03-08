# 🧠 InsightX AI V2: Universal RAG & Data Analyst

**InsightX AI** is a production-grade, multi-page Retrieval-Augmented Generation (RAG) platform. It allows users to chat with any file—documents, spreadsheets, code, or images—instantly with near-zero latency.

## 🚀 Key Features (Version 2.0)

*   **💬 Semantic Chat**: Dedicated RAG workspace for conversational AI. No heavy model downloads required.
*   **⚡ Precise Calculator**: Code-driven mathematical engine that runs real Python/Pandas on 100% of your data.
*   **📊 Leadership Dashboard**: Automated visual insights and executive metrics detection.
*   **👀 Data Audit**: Full statistical summaries, duplicate detection, and raw data inspection.
*   **🛡️ Performance Engine**: Memory-resident numeric downcasting and hybrid TF-IDF indexing for extreme scale (250k+ rows).

## 🏗️ Architecture

InsightX V2 uses a modular **Multi-Page Architecture**:
- **`app.py`**: The Home Page and Data Ingestion Hub.
- **`utils.py`**: The Central Engine (Shared extraction, RAG, and AI logic).
- **`pages/`**: Specialized workspaces for Chat, Calculations, and Dashboards.

## 🛠️ Tech Stack

*   **Frontend:** [Streamlit](https://streamlit.io/) (Multi-page configuration)
*   **LLM:** [Groq LPU](https://groq.com/) (Meta Llama 3.3 70B)
*   **Data Processing:** Pandas (with numeric downcasting), NumPy
*   **Search/RAG:** Scikit-learn (TF-IDF Vectorization)
*   **OCR:** Tesseract + Pypdf + Pdf2image

## ⚙️ Setup & Installation

### 1. Prerequisites
Ensure you have Python 3.9+ and system libraries for OCR:
- **Windows:** [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) and [Poppler](https://github.com/oschwartz10612/poppler-windows/releases/).
- **Linux:** `sudo apt-get install tesseract-ocr poppler-utils`

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Set your Streamlit secrets or `.env`:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the App
```bash
streamlit run app.py
```

---
*Developed for the Hackathon V2 Upgrade.*
