# ==============================================
# 🧠 Llama3.2 + HuggingFace Embeddings + RAG Chatbot
# ==============================================

import os
import glob
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
import gradio as gr


# ==============================================
# 🔧 Step 1: Environment Setup
# ==============================================

# Ensure Ollama is running locally with llama3.2
# Command: ollama run llama3.2
os.environ["OPENAI_API_KEY"] = "ollama"
os.environ["OPENAI_BASE_URL"] = "http://localhost:11434/v1"

# Folder where documents are stored
base_dir = "knowledge-base"   # You can change this


# ==============================================
# 📂 Step 2: Load Documents Dynamically
# ==============================================

def load_documents(base_dir):
    documents = []
    folders = glob.glob(f"{base_dir}/*")

    for folder in folders:
        pdfs = glob.glob(f"{folder}/*.pdf")
        docs = glob.glob(f"{folder}/*.docx")
        txts = glob.glob(f"{folder}/*.txt")
        mds = glob.glob(f"{folder}/*.md")

        for path in pdfs:
            documents.extend(PyPDFLoader(path).load())
        for path in docs:
            documents.extend(Docx2txtLoader(path).load())
        for path in txts:
            documents.extend(TextLoader(path).load())
        for path in mds:
            documents.extend(UnstructuredMarkdownLoader(path).load())

    print(f"✅ Loaded {len(documents)} total documents.")
    return documents


# ==============================================
# ✂️ Step 3: Split Documents into Chunks
# ==============================================

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks.")
    return chunks


# ==============================================
# 🧬 Step 4: Create and Persist Vector Database
# ==============================================

def create_vector_db(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db_name = "vector_db"
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory=db_name)
    vectorstore.persist()
    print(f"✅ Vector DB created and persisted at '{db_name}'.")
    return vectorstore


# ==============================================
# 🦙 Step 5: Create Llama3.2-based RAG Chain
# ==============================================

def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    llm = OpenAI(model="llama3.2", base_url="http://localhost:11434/v1", api_key="ollama")

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    return qa_chain


# ==============================================
# 💬 Step 6: Define Gradio Chat Interface
# ==============================================

def launch_gradio_chat(qa_chain):
    def chat_fn(message, history=[]):
        response = qa_chain({"question": message})
        return response["answer"]

    iface = gr.ChatInterface(
        fn=chat_fn,
        title="💬 Llama3.2 RAG Chatbot",
        description="Ask questions based on your uploaded documents (PDF, DOCX, TXT, MD)",
    )

    iface.launch(server_name="0.0.0.0", server_port=7860)


# ==============================================
# 🚀 Step 7: Main Execution Flow
# ==============================================

if __name__ == "__main__":
    docs = load_documents(base_dir)
    chunks = split_documents(docs)
    vectordb = create_vector_db(chunks)
    qa_chain = create_qa_chain(vectordb)
    launch_gradio_chat(qa_chain)
