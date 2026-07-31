# 📄 Chat with Your PDF (RAG Application)

An interactive, AI-powered Retrieval-Augmented Generation (RAG) application built to extract precise insights, summaries, and context-driven answers directly from PDF documents in real-time.

---
📽️ **[Live App is deployed! Try it here:](https://giridhar-rag-pdf-chat.streamlit.app/)**

## ✨ Features

* **⚡ Fast Vector Embedding:** Ingests and processes complex PDFs seamlessly into vector representations.
* **🔍 Contextual QA & Summarization:** Ask dynamic questions or request targeted summaries (e.g., bulleted highlights, technical skill breakdowns).
* **🎯 High-Precision Answers:** Employs RAG techniques to eliminate hallucinations and constrain model responses strictly to source text.
* **🕵️ Inspection Capabilities:** Includes a **"Show Retrieved Context"** drawer to inspect exact chunks feeding into the generation step.
* **🖥️ Interactive UI:** Built with Streamlit for a clean, user-friendly interface.

---

## 🛠️ Tech Stack & Tools

* **Frontend / UI:** Streamlit
* **Language / Frameworks:** Python, PyTorch, LangChain / LlamaIndex *(adapt based on your setup)*
* **Embeddings & Vector Store:** Hugging Face Embeddings / FAISS / Chroma DB *(adapt based on your setup)*
* **LLM Engine:** OpenAI / Gemini / Ollama / Local Models

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure Python 3.9+ is installed on your system.

### 2. Installation
```bash
# Clone the repository
git clone [https://github.com/your-username/rag-pdf-chatbot.git](https://github.com/your-username/rag-pdf-chatbot.git)
cd rag-pdf-chatbot

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the App
```bash
streamlit run app.py
```

---

## 📸 Demo Workflow

1. **Upload Document:** Load any PDF up to 200MB.
2. **Auto-Embedding:** System processes text chunks and generates embeddings automatically.
3. **Query & Inspect:** Ask questions in natural language and expand the context drawer to verify vector-matched passages.

---

## 🤝 Key Highlights for Engineering Roles

This project demonstrates practical implementation of:
* Document preprocessing, chunking, and metadata parsing.
* Vector search indexing and similarity retrieval pipelines.
* Grounded prompt engineering and multi-turn context retention.
