# 🧪 PaperPocket

PaperPocket is a lightweight research assistant.  
Upload a PDF, summarize it, and ask questions about its content using an interchangeable LLM backend.

---

## Features
- PDF upload & text extraction  
- Text chunking for long documents  
- Automatic summarization (≤10 bullet points)  
- Retrieval-augmented QA (answers grounded in context)  
- Streamlit interface  

---

## ⚙️ Installation

```bash
git clone https://github.com/rabieessayeh/PaperPocket.git
cd PaperPocket
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows
pip install -r requirements.txt 
```



## ⚙️ Configuration

PaperPocket uses environment variables (from `.env` if present).

### 🔹 Ollama
- `OLLAMA_HOST` (default: `http://localhost:11434`)  
- `OLLAMA_MODEL` (e.g., `llama3.1:8b-instruct-q4_K_M`)  
- Optional: `OLLAMA_NUM_CTX`, `OLLAMA_NUM_PREDICT`, `OLLAMA_NUM_THREAD`, `OLLAMA_KEEP_ALIVE`, `OLLAMA_HTTP_TIMEOUT`  

### 🔹 GPT-OSS-20B
- `GPTOSS_HOST` (default: `http://localhost:8000/v1`)  
- `GPTOSS_MODEL` (default: `gpt-oss-20b`)  
- `GPTOSS_API_KEY` (token if required)  
- `GPTOSS_TIMEOUT` (default: `300`)  

### 🔹 Embeddings
- `EMBED_MODEL` (default: `intfloat/multilingual-e5-small`)  
- `EMBED_DEVICE` (default: auto)  
- `EMBED_DTYPE` (default: `float16`)  

### Example `.env`
```env
# --- Ollama ---
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b-instruct-q4_K_M

# --- GPT-OSS-20B ---
# GPTOSS_HOST=http://localhost:8000/v1
# GPTOSS_MODEL=gpt-oss-20b
# GPTOSS_API_KEY=your_api_key_here

# --- Embeddings ---
EMBED_MODEL=intfloat/multilingual-e5-base

```
---

## ▶️ Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.


---

## 🧩 How It Works

1. **PDF parsing** → `extract_text` reads content  
2. **Text preprocessing** → `chunk_text` splits into overlapping chunks  
3. **Indexing** → `prepare_retriever` builds dense vectors with embeddings  
4. **Summarization** → map-reduce style, concise factual bullets  
5. **QA** → retrieve relevant passages + answer strictly from context  
6. **LLM backend** → Ollama provides completions  

---

## 📜 License

MIT License – feel free to use, modify, and share.  
