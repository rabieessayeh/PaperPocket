# 🧪 PaperPocket

PaperPocket is a lightweight web app that helps researchers interact with scientific papers.  
You can **upload a PDF**, **summarize it automatically**, and **ask questions** about its content using an AI assistant powered by **Ollama** and local embeddings.

---

## 🚀 Features

- 📄 **PDF Upload & Parsing** – extract text from research articles.  
- ✂️ **Text Chunking** – split long documents into overlapping segments.  
- 📝 **Summarization** – generate concise factual summaries (≤10 bullet points).  
- 🔍 **Retriever-Augmented QA** – ask questions about the paper, answers are grounded in retrieved context.  
- ⚡ **Local Embeddings** – optional use of `sentence-transformers` or custom embedding models.  
- 🖥️ **Streamlit UI** – simple, interactive interface.  

---

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/paperpocket.git
cd paperpocket
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

Dependencies include:
- **ML & Embeddings**: PyTorch, Transformers, Sentence-Transformers, FAISS  
- **PDF Processing**: PyPDF2, pdfplumber  
- **Web App**: Streamlit, python-dotenv, Requests  
- **Utils**: NumPy, scikit-learn  

---

## ⚙️ Configuration

PaperPocket uses environment variables (from `.env` if present):

- **Ollama settings**:  
  - `OLLAMA_HOST` (default: `http://localhost:11434`)  
  - `OLLAMA_MODEL` (default: `llama3.1:latest`)  
  - `OLLAMA_NUM_CTX`, `OLLAMA_NUM_PREDICT`, `OLLAMA_NUM_THREAD`, etc.

- **Embedding model settings**:  
  - `EMBED_MODEL` (default: `intfloat/multilingual-e5-small`)  
  - `EMBED_DEVICE` (default: auto)  
  - `EMBED_DTYPE` (default: `float16`)  

Example `.env` file:
```env
OLLAMA_MODEL=llama3.1:8b-instruct-q4_K_M
EMBED_MODEL=intfloat/multilingual-e5-base
```

---

## ▶️ Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

### Workflow
1. **Upload a PDF** in the sidebar.  
2. The document is automatically **indexed** for retrieval.  
3. Use **Summarize** to get a concise overview.  
4. Ask your own questions in natural language, and the assistant will answer based only on the paper’s content.  

---

## 📂 Project Structure

```
.
├── app.py              # Streamlit UI
├── core/
│   ├── pdf_reader.py   # PDF text extraction
│   ├── text_utils.py   # Cleaning & chunking
│   ├── summarize.py    # Summarization pipeline
│   ├── qa.py           # Retriever & QA
│   ├── model_ollama.py # Ollama client
│   └── model_local.py  # custom embeddings
├── requirements.txt
└── README.md
```

---

## 🧩 How It Works

1. **PDF parsing** → `extract_text` reads content  
2. **Text preprocessing** → `chunk_text` splits into overlapping chunks  
3. **Indexing** → `prepare_retriever` builds dense vectors with embeddings  
4. **Summarization** → map-reduce style, concise factual bullets  
5. **QA** → retrieve relevant passages + answer strictly from context  
6. **LLM backend** → Ollama provides completions  

---

## 🧑‍💻 Requirements

- Python **3.9+**  
- [Ollama](https://ollama.ai) installed and running locally  
- (Optional) GPU for faster embeddings  

---

## 📜 License

MIT License – feel free to use, modify, and share.  
