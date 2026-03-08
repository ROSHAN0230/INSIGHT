# 💎 Lumina AI: Supreme Data Intelligence

**Lumina AI** is a production-grade, multi-page Retrieval-Augmented Generation (RAG) platform with an elite 'Chrome Abstract' user experience. It allows users to chat with any file—documents, spreadsheets, code, or images—instantly with an immersive 3D interface.

## 🚀 Key Features
*   **💬 Semantic Chat**: Dedicated RAG workspace for conversational AI.
*   **⚡ Precise Logic Engine**: Code-driven mathematical engine that runs real Python/Pandas on 100% of your data.
*   **📊 Visual Intelligence**: Automated visual insights and executive metrics detection.
*   **👀 Data Audit**: Full statistical summaries, duplicate detection, and raw data inspection.
*   **💎 Chrome Abstract UI**: A high-end 3D visual language with glassmorphism and neon glows.

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
