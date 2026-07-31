import os
import streamlit as st
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai

# Page Config
st.set_page_config(page_title="AI Chat with PDF", page_icon="📄")
st.title("📄 Chat with your PDF (RAG Application)")

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("Please set your GEMINI_API_KEY in the .env file.")
    st.stop()

# Initialize Gemini Client
genai_client = genai.Client(api_key=api_key)

# Sidebar for PDF Upload
st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Save PDF temporarily
    temp_pdf_path = f"temp_{uploaded_file.name}"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.success("PDF Uploaded Successfully!")

    # Process PDF and Store in ChromaDB
    with st.spinner("Processing PDF and creating Vector Embeddings..."):
        loader = PyPDFLoader(temp_pdf_path)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(pages)

        # ChromaDB setup
    client = chromadb.Client()
    default_ef = embedding_functions.DefaultEmbeddingFunction()

    collection = client.get_or_create_collection(
        name="ui_pdf_knowledge_base",
        embedding_function=default_ef
    )
    documents_list = [chunk.page_content for chunk in chunks]
    ids_list = [f"ui_id_{i}" for i in range(len(chunks))]

    collection.add(documents=documents_list, ids=ids_list)

    st.success("PDF processed and ready for questions!")

    # Chat Interface
    user_query = st.text_input("Ask a question about your PDF:")

    if user_query:
        with st.spinner("Searching and generating answer..."):
            # Retrieval
            search_results = collection.query(query_texts=[user_query], n_results=3)
            retrieved_context = "\n---\n".join(search_results['documents'][0])

            # Prompt
            prompt = f"""
            You are a helpful AI Assistant. Answer the user's question based ONLY on the provided context below.
            If the answer cannot be found in the context, politely say "I cannot find this information in the provided document."

            --- CONTEXT FROM PDF ---
            {retrieved_context}

            --- USER QUESTION ---
            {user_query}

            --- ANSWER ---
            """

            response = genai_client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt
            )

            st.markdown("### 🤖 Answer:")
            st.write(response.text)

            with st.expander("Show Retrieved Context (RAG Inspection)"):
                st.write(retrieved_context)