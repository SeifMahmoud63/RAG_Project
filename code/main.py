from Agent import build_agent
from Vector_db import vector_db
from chunking import chunking
from loader import load_documents_from_folder
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from retriever import advanced_retrieve
import os


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

config = {"configurable": {"thread_id": "seif_session"}, "recursion_limit": 10}

print("--- Chat Started (Type 'exit' to stop) ---")

while True:
    query = input("\nUser: ")
    if query.lower() in ['exit', 'quit']:
        break

    input_data = {"messages": [HumanMessage(content=query)]}
    
    result = agent_app.invoke(input_data, config=config)
    
    print(f"\nAI: {result['messages'][-1].content}")




