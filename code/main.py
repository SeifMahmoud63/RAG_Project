import os
from dotenv import load_dotenv
from loader import load_documents_from_folder
from chunking import chunking
from Vector_db import vector_db
from retriever import advanced_retrieve
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def main():
    data_path = "../data/"
    docs = load_documents_from_folder(data_path)

    if not docs:
        print("No documents found in the specified folder.")
        return

    chunks = chunking(docs)
    vector = vector_db(chunks)

    query = input("\n Enter your question ?")
    if not query.strip():
        print("Empty query")
        return

    results = advanced_retrieve(vector, chunks, query)

    llm = ChatGroq(model="llama-3.1-8b-instant")

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a professional academic assistant. "
            "Use ONLY the provided context to answer the user's question. "
            "If the answer is not contained within the context, strictly say: "
            "'I'm sorry, but the provided documents do not contain information about this.' "
            "Do not use outside knowledge or hallucinate."
        )),
        ("human", "Context:\n{context}\n\nQuestion: {input}")
    ])


    context_text = "\n\n".join([doc.page_content for doc in results])
    
    chain = prompt_template | llm
    
    response = chain.invoke({
        "context": context_text,
        "input": query
    })

    print("Answer:\n", response.content)

if __name__ == "__main__":
    main()


