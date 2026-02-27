from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.retrievers.tavily_search_api import TavilySearchAPIRetriever
from langchain_core.tools import tool
from dotenv import load_dotenv
load_dotenv()
from langchain_cohere import ChatCohere

import os
COHERE_API_KEY=os.getenv("COHERE_API_KEY")

def get_llm(temperature=0):
    return ChatCohere(
        model="command-a-03-2025",
        temperature=temperature
    )


def get_embedding():
    return HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")