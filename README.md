# 🧠 InsightX AI: Universal RAG & Data Analyst

**InsightX AI** is a lightning-fast, privacy-conscious Retrieval-Augmented Generation (RAG) tool that allows users to chat with any file—documents, spreadsheets, code, or images—instantly.

## 🚀 Key Features

*   **💬 Ask AI (RAG):** Smart TF-IDF indexing for instant Q&A on any file. No heavy model downloads required.
*   **⚡ Precise Calculator:** Runs real Python/pandas code on your full dataset for 100% exact numerical answers (totals, averages, trends).
*   **📊 Data Analysis:** Automated statistical summaries, data quality checks, and interactive charts (Bar, Line, Area).
*   **👀 Universal Preview:** Supports CSV, Excel, PDF (with OCR), Word, PPT, JSON, ZIP, Images, and more.
*   **📂 Multi-format Extraction:** Instant text extraction from almost any file type.

## 🛠️ Tech Stack

*   **Frontend:** [Streamlit](https://streamlit.io/)
*   **LLM:** [Groq](https://groq.com/) (Llama 3.3 70B Versatile)
*   **Data Processing:** Pandas, NumPy
*   **Search/RAG:** Scikit-learn (TF-IDF Vectorization)
*   **OCR:** Tesseract, Pypdf, Pdf2image

## ⚙️ Setup & Installation

### 1. Prerequisites
Ensure you have Python 3.9+ installed and specialized system libraries for OCR:
- **Windows:** Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) and [Poppler](https://github.com/oschwartz10612/poppler-windows/releases/).
- **Linux:** `sudo apt-get install tesseract-ocr poppler-utils`

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file or set your Streamlit secrets with:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the App
```bash
streamlit run app.py
```

## 📝 Usage Tip
- For **summaries and explanations**, use the "Ask AI" tab.
- For **exact calculations and charts**, use the "Precise Calculator" or "Data Analysis" tabs.

---
*Built for the Hackathon Submission.*
