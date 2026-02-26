from Agent import build_agent
from Vector_db import vector_db
from chunking import chunking
from loader import load_documents_from_folder
from langchain_core.messages import HumanMessage
import os
from langchain_core.documents import Document

persist_directory = "chroma_db"
if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
    docs = load_documents_from_folder("../data/")
    chunks = chunking(docs)
    vector = vector_db(chunks)
else:
    vector = vector_db()
    db_data = vector.get(include=["documents", "metadatas"])
    chunks = [
        Document(page_content=db_data["documents"][i],
                 metadata=db_data["metadatas"][i] if db_data["metadatas"] else {})
        for i in range(len(db_data["documents"]))
    ]

agent_app = build_agent(vector, chunks)

query = input("Enter your question: ")
messages = [HumanMessage(content=query)]
result = agent_app.invoke({"messages": messages})
print(result["messages"][-1].content)