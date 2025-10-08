import os
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader, 
    Docx2txtLoader, 
    TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma



EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DB_DIR = "vector_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200



def load_documents(input_paths):
    """
    Recursively loads PDF, DOCX, MD, TXT from one or more folders/files.
    Args:
        input_paths (list[str]): list of directories or file paths
    Returns:
        list[Document]
    """
    supported_exts = [".pdf", ".docx", ".md", ".txt"]
    documents = []

    for path_str in input_paths:
        path = Path(path_str)
        if not path.exists():
            print(f"⚠️ Skipping missing path: {path}")
            continue

        files = [path] if path.is_file() else list(path.rglob("*"))
        for file in files:
            if file.suffix.lower() not in supported_exts:
                continue

            if file.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file))
            elif file.suffix.lower() == ".docx":
                loader = Docx2txtLoader(str(file))
            else:
                loader = TextLoader(str(file), encoding="utf-8")

            docs = loader.load()
            for d in docs:
                d.metadata["source_file"] = str(file)
                d.metadata["file_type"] = file.suffix.lower()
                d.metadata["parent_dir"] = file.parent.name
                documents.append(d)

    print(f"Loaded {len(documents)} documents from {len(input_paths)} input path(s)")
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def create_embeddings(chunks, persist_dir=DB_DIR):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    os.makedirs(persist_dir, exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    count = vectorstore._collection.count()
    print(f"✅ Vectorstore created with {count} embedded chunks at: {persist_dir}")
    return vectorstore


if __name__ == "__main__":
    # You can dynamically define any folder(s) 
    INPUT_PATHS = [
        r"C:\U**rs\s**hm\Documents\project_docs",
        r"D:\ResearchPapers",
        "manuals"  # relative folder also fine
    ]

    docs = load_documents(INPUT_PATHS)
    chunks = split_documents(docs)
    vectorstore = create_embeddings(chunks)

    sample = chunks[0]
    print("\n Sample Chunk Preview:")
    print("Text:", sample.page_content[:200], "...")
    print("Metadata:", sample.metadata)
