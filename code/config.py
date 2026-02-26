from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

def get_llm(temperature=0):
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=temperature
    )


def get_embedding():
    return HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")