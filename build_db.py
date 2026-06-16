import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Load your secure API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

print("Loading the induction manual...")

# 1. Load the PDF document from your data folder
loader = PyPDFLoader("data/manual.pdf")
documents = loader.load()

# 2. Split the document into small, readable chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

print(f"Manual successfully split into {len(chunks)} technical chunks.")
print("Building the vector database. This may take a moment...")

# 3. Convert chunks into vector embeddings and save to ChromaDB
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
)

print("Database built and secured. Ready for diagnostic retrieval.")