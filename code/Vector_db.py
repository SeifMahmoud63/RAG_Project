# from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# from langchain_community.vectorstores import Chroma

# def vector_db(docs):
#     emb=HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vector=Chroma.from_documents(
#         documents=docs,embedding=emb,
#         collection_name="Machine_Learning"
#         ,persist_directory="chroma_db"
#     )
#     return vector

import os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

def vector_db(docs=None):
    embedding_model = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    persist_directory = "chroma_db"
    collection_name = "Machine_Learning"

    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print(f"--- Loading existing Vector Store from: {persist_directory} ---")
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model,
            collection_name=collection_name
        )
    else:
        print("--- No existing Vector Store found. Creating a new one... ---")
        if docs is None:
            raise ValueError("The 'docs' parameter is required to initialize the database for the first time.")
            
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
    
    return vectorstore

