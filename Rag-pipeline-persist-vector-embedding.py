

import os
import gradio as gr
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

load_dotenv()
DB_DIR = "vector_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"  # Model name accessible via your OpenAI-compatible endpoint


print("🔹 Loading embeddings and Chroma DB...")
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
print(f"Loaded Chroma DB with {vectorstore._collection.count()} chunks")


print("Connecting to Llama 3.2 model using OpenAI API key...")

llm = ChatOpenAI(
    model=LLM_MODEL,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.3,
    max_tokens=512
)


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)


def chat(query, history):
    if not query.strip():
        return history + [("Please enter a question.", "")]
    try:
        result = qa_chain(query)
        answer = result["result"]
        sources = "\n".join(
            [f"- {doc.metadata.get('source', 'unknown')}" for doc in result["source_documents"]]
        )
        response = f"{answer}\n\n📚 **Sources:**\n{sources}"
        history.append((query, response))
        return history
    except Exception as e:
        history.append((query, f"❌ Error: {str(e)}"))
        return history


with gr.Blocks(title="Llama 3.2 RAG Chatbot") as demo:
    gr.Markdown("## 🧠 Llama 3.2 RAG Chatbot\nAsk questions from your document database!")
    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(label="Ask a question...")
    clear = gr.Button("Clear Chat")

    msg.submit(chat, [msg, chatbot], [chatbot])
    clear.click(lambda: [], None, chatbot, queue=False)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
