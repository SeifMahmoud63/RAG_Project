from langchain_core.tools import tool
from langchain_community.retrievers.tavily_search_api import TavilySearchAPIRetriever
from retriever import advanced_retrieve 

tavily = TavilySearchAPIRetriever(k=3)

def create_tools(vector, chunks):
    
    @tool
    def Search_Local_Documents(query: str) -> str:
        """Use this tool to search for technical information about Data Science, ML, KNN, and Ensemble Learning from the uploaded PDF/PPTX files."""
        results = advanced_retrieve(vector, chunks, query)
        return "\n\n".join([doc.page_content for doc in results]) if results else "No info found in documents."

    @tool
    def Tavily_Tool(query: str) -> str:
        """Use this tool to search the internet for general knowledge, current events, or things not found in local documents."""
        docs = tavily.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs])

    return [Search_Local_Documents, Tavily_Tool]