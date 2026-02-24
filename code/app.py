import os
import streamlit as st
from loader import load_documents_from_folder
from chunking import chunking
from Vector_db import vector_db
from retriever import advanced_retrieve
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

st.set_page_config(
    page_title="RAG Chat Assistant",
    layout="wide"
)

st.title("RAG Chat Assistant")


@st.cache_resource
def load_rag_system():

    persist_directory = "chroma_db"
    data_path = "../data/"

    chunks = None

    if not os.path.isdir(persist_directory) or not os.listdir(persist_directory):

        docs = load_documents_from_folder(data_path)

        if not docs:
            st.error("No documents found!")
            return None, None

        chunks = chunking(docs)
        vector = vector_db(chunks)

    else:

        vector = vector_db()

        db_data = vector.get(include=["documents", "metadatas"])

        chunks = [
            Document(page_content=db_data["documents"][i])
            for i in range(len(db_data["documents"]))
        ]

    return vector, chunks


vector_store, chunks = load_rag_system()

llm = ChatGroq(model="llama-3.1-8b-instant")

prompt_template = ChatPromptTemplate.from_messages([
    ("system",
     "You are a professional academic assistant. "
     "Use ONLY provided context to answer. "
     "If answer is not found say: "
     "'I don't have enough information in the documents.'"),

    ("human", "Context:\n{context}\n\nQuestion:\n{input}")
])

chain = prompt_template | llm

query = st.text_input("Ask your question:")

if st.button("Send"):

    if not query:
        st.warning("Please enter a question")
        st.stop()

    if chunks is None:
        st.error("Chunks not loaded")
        st.stop()

    results = advanced_retrieve(vector_store, chunks, query)

    context_text = "\n\n".join(
        [doc.page_content for doc in results]
    )

    response = chain.invoke({
        "context": context_text,
        "input": query
    })

    st.success("Answer:")
    st.write(response.content)