import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Modern Integration Imports (Zero Warnings)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def load_and_vectorize_pdf(pdf_path: str):
    print(f"📖 Loading offline investing manual: {pdf_path}...")
    if not os.path.exists(pdf_path):
        print(f"❌ Error: Could not find '{pdf_path}'. Place the file in this directory.")
        return

    # 1. Parse the PDF pages completely offline
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # 2. Break down dense chapters into digestible paragraphs
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    print(f"✂️ Split book into {len(chunks)} individual semantic chunks.")
    
    # 3. Initialize the updated, stable embedding engine locally
    print("🧠 Initializing local embedding engine (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 4. Spin up the modern standalone Chroma storage
    print("💾 Saving embeddings to local disk ('chroma_db' directory)...")
    Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
    print("✅ Offline Knowledge Base successfully built!")

if __name__ == "__main__":
    load_and_vectorize_pdf("investing_manual.pdf")