from langchain_groq import ChatGroq
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_compressors import FlashrankRerank
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="llama-3.1-8b-instant")

# Query Rewriting
def rewrite_query(query):
    prompt = f"""
    Rewrite the following question to be more specific and optimized for document retrieval:

    Question: {query}
    """
    return llm.invoke(prompt).content


# HyDE Generation
def generate_hyde(query):
    prompt = f"""
    Write a detailed answer to the following question:

    Question: {query}
    """
    return llm.invoke(prompt).content

def hybrid_search(vector_store, docs, query):

    # Vector (MMR search)
    vector_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4}
    )
    vector_docs = vector_retriever.invoke(query)

    # BM25
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 4
    bm25_docs = bm25_retriever.invoke(query)


    return list({doc.page_content: doc for doc in vector_docs + bm25_docs}.values())


# Reranking
def rerank(query, documents, top_k=4):
    compressor = FlashrankRerank(
        top_n=top_k
    )

    return compressor.compress_documents(
        documents=documents,
        query=query
    )


# Final Advanced Retrieval
def advanced_retrieve(vector_store, docs, query):

    rewritten_query = rewrite_query(query)

 
    hyde_doc = generate_hyde(rewritten_query)

    candidates = hybrid_search(vector_store, docs, hyde_doc)

    final_docs = rerank(rewritten_query, candidates)

    return final_docs