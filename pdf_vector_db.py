import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ Error: GEMINI_API_KEY not found in .env file!")
    exit()

pdf_path = "sample.pdf"

if not os.path.exists(pdf_path):
    print(f"❌ Error: {pdf_path} file not found in folder!")
    exit()

print("📄 Loading PDF content...")
# 2. Load PDF file
loader = PyPDFLoader(pdf_path)
pages = loader.load()

# 3. Split PDF text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(pages)

print(f"✂️ Split PDF into {len(chunks)} chunks.")

# 4. Store chunks into ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
default_ef = embedding_functions.DefaultEmbeddingFunction()

collection = chroma_client.get_or_create_collection(
    name="pdf_knowledge_base",
    embedding_function=default_ef
)

# Prepare documents and IDs for insertion
documents_list = [chunk.page_content for chunk in chunks]
ids_list = [f"id_{i}" for i in range(len(chunks))]

print("📥 Storing PDF chunks into ChromaDB...")
collection.add(
    documents=documents_list,
    ids=ids_list
)
print("✅ PDF processing complete & saved in Vector DB!\n")

# 5. Perform Vector Search on the PDF
query_text = "What is the main topic of this document?"
print(f"🔍 Searching PDF for: '{query_text}'...")

results = collection.query(
    query_texts=[query_text],
    n_results=1
)

print("\n🎯 Search Result from PDF:")
print("Found Match:", results['documents'][0][0])