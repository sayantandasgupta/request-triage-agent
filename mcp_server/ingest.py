import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
ENV_PATH = PROJECT_ROOT / ".env"
DOCS_DIR = BASE_DIR / "documents"
CHROMA_DIR = BASE_DIR / "chroma_db"

load_dotenv(ENV_PATH)


def ingest():
    if CHROMA_DIR.exists() and CHROMA_DIR.is_dir():
        shutil.rmtree(CHROMA_DIR)
    
    
    policy_docs = []
    
    for pdf_path in sorted(DOCS_DIR.glob("*.pdf")):
        data = PyPDFLoader(str(pdf_path)).load()
        for d in data:
            d.metadata["policy_id"] = d.metadata["subject"]
            d.metadata["filename"] = pdf_path.name
        policy_docs.extend(data)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    docs = splitter.split_documents(policy_docs)

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(f"OPENAI_API_KEY is not set. Add it to {ENV_PATH}")
 
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    Chroma.from_documents(
        docs,
        embedding=embedding,
        collection_name="policies",
        persist_directory=str(CHROMA_DIR),
    )


if __name__ == "__main__":
    ingest()
