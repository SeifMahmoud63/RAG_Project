from langchain_community.vectorstores import Chroma
import os
from config import get_embedding
def vector_db(docs=None):
    embedding_model =get_embedding()
    persist_directory = "chroma_db"
    collection_name = "Machine_Learning"

    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        # Just load it
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model,
            collection_name=collection_name
        )
    else:
        if docs is None:
            raise ValueError("Docs required for first initialization")
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
    return vectorstore
