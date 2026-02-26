import streamlit as st
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

from Agent import build_agent
from Vector_db import vector_db
from chunking import chunking
from loader import load_documents_from_folder

st.set_page_config(page_title="Seif AI Agent", page_icon="🧠🇦🇮")
st.title("RAG Multi-Tool Agent")

@st.cache_resource
def initialize_system():
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
    return agent_app

agent_app = initialize_system()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything about your files or the web..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        config = {"configurable": {"thread_id": "streamlit_session"}, "recursion_limit": 10}
        
        try:
            input_data = {"messages": [HumanMessage(content=prompt)]}
            result = agent_app.invoke(input_data, config=config)
            
            final_answer = result["messages"][-1].content
            
            message_placeholder.markdown(final_answer)
            
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

with st.sidebar:
    st.info("Files loaded from '../data/' folder.")
    if st.button("Clear Chat Memory"):
        st.session_state.messages = []
        st.rerun()