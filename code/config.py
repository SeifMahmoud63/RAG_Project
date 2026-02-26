from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.retrievers.tavily_search_api import TavilySearchAPIRetriever
from langchain_core.tools import tool

def get_llm(temperature=0):
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=temperature
    )


def get_embedding():
    return HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


tavily = TavilySearchAPIRetriever(k=3)
@tool
def tavily_search(query: str) -> str:
    """Search documents using Tavily API and return top 3 results."""
    docs = tavily.get_relevant_documents(query)
    return "\n\n".join([doc.page_content for doc in docs])

def get_tools():
    return [tavily_search]