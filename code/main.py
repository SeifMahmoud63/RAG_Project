import os
from dotenv import load_dotenv
from loader import load_documents_from_folder
from chunking import chunking
from Vector_db import vector_db
from retriever import advanced_retrieve
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

load_dotenv()


def main():

    persist_directory = "chroma_db" 

    if not os.path.exists(persist_directory) or not os.listdir(persist_directory):

        data_path = "../data/"

        docs = load_documents_from_folder(data_path)

        if not docs:
            print("No documents found.")
            return

        chunks = chunking(docs)
        vector = vector_db(chunks)

    else:
        vector = vector_db()

        db_data = vector.get(include=["documents", "metadatas"])

        chunks = [
            Document(
                page_content=db_data["documents"][i],
                metadata=db_data["metadatas"][i] if db_data["metadatas"] else {}
            )
            for i in range(len(db_data["documents"]))
        ]


    llm = ChatGroq(model="llama-3.1-8b-instant")

    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
         "You are a professional academic assistant. "
         "Use ONLY the provided context to answer the user's question. "
         "If the answer is not in context say exactly: "
         "'I'm sorry, but the provided documents do not contain information about this.'"
         ),
        ("human", "Context:\n{context}\n\nQuestion: {input}")
    ])

    chain = prompt_template | llm

    query = input("\nEnter your question: ").strip()

    results = advanced_retrieve(vector, chunks, query)

    context_text = "\n\n".join(
            [doc.page_content for doc in results]
        )

    response = chain.invoke({
            "context": context_text,
            "input": query
        })

    print("\n Answer:\n", response.content)


if __name__ == "__main__":
    main()