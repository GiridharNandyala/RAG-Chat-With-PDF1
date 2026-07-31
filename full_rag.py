import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai

# 1. Environment & API Key Setup
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ Error: GEMINI_API_KEY not found in .env file!")
    exit()

# Gemini SDK Client Initialization
client = genai.Client(api_key=api_key)

pdf_path = "sample.pdf"

if not os.path.exists(pdf_path):
    print(f"❌ Error: {pdf_path} file not found!")
    exit()

print("📄 Loading and processing PDF...")
# 2. Load PDF & Split into Chunks
loader = PyPDFLoader(pdf_path)
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(pages)

# 3. Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
default_ef = embedding_functions.DefaultEmbeddingFunction()

collection = chroma_client.get_or_create_collection(
    name="full_rag_knowledge_base",
    embedding_function=default_ef
)

documents_list = [chunk.page_content for chunk in chunks]
ids_list = [f"rag_id_{i}" for i in range(len(chunks))]

collection.add(
    documents=documents_list,
    ids=ids_list
)
print("✅ PDF successfully embedded and saved in Vector DB!\n")

# 4. User Query Definition
user_query = "Summarize the key technical skills and projects mentioned in this document."
print(f"❓ User Question: '{user_query}'\n")

print("🔍 Retrieving relevant context from ChromaDB...")
search_results = collection.query(
    query_texts=[user_query],
    n_results=3  # Top 3 relevant chunks
)

# Retrieved text chunks
retrieved_context = "\n---\n".join(search_results['documents'][0])

# 5. RAG Prompt Engineering for Gemini
prompt = f"""
You are a helpful AI Assistant. Answer the user's question based ONLY on the provided context below.
If the answer cannot be found in the context, politely say "I cannot find this information in the provided document."

--- CONTEXT FROM PDF ---
{retrieved_context}

--- USER QUESTION ---
{user_query}

--- ANSWER ---
"""

print("🤖 Generating answer using Gemini API...")
# 6. Generate Response using Gemini API
response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=prompt
)

print("\n✨ Gemini RAG Response:")
print("="*50)
print(response.text)
print("="*50)