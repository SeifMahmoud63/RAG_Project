# import os
# import streamlit as st
# from loader import load_documents_from_folder
# from chunking import chunking
# from Vector_db import vector_db
# from retriever import advanced_retrieve
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.documents import Document

# st.set_page_config(
#     page_title="RAG Chat Assistant",
#     layout="wide"
# )

# st.title("RAG Chat Assistant")


# @st.cache_resource
# def load_rag_system():

#     persist_directory = "chroma_db"
#     data_path = "../data/"

#     chunks = None

#     if not os.path.isdir(persist_directory) or not os.listdir(persist_directory):

#         docs = load_documents_from_folder(data_path)

#         if not docs:
#             st.error("No documents found!")
#             return None, None

#         chunks = chunking(docs)
#         vector = vector_db(chunks)

#     else:

#         vector = vector_db()

#         db_data = vector.get(include=["documents", "metadatas"])

#         chunks = [
#             Document(page_content=db_data["documents"][i])
#             for i in range(len(db_data["documents"]))
#         ]

#     return vector, chunks


# vector_store, chunks = load_rag_system()

# llm = ChatGroq(model="llama-3.1-8b-instant")

# prompt_template = ChatPromptTemplate.from_messages([
#     ("system",
#      "You are a professional academic assistant. "
#      "Use ONLY provided context to answer. "
#      "If answer is not found say: "
#      "'I don't have enough information in the documents.'"),

#     ("human", "Context:\n{context}\n\nQuestion:\n{input}")
# ])

# chain = prompt_template | llm

# query = st.text_input("Ask your question:")

# if st.button("Send"):

#     if not query:
#         st.warning("Please enter a question")
#         st.stop()

#     if chunks is None:
#         st.error("Chunks not loaded")
#         st.stop()

#     results = advanced_retrieve(vector_store, chunks, query)

#     context_text = "\n\n".join(
#         [doc.page_content for doc in results]
#     )

#     response = chain.invoke({
#         "context": context_text,
#         "input": query
#     })

#     st.success("Answer:")
#     st.write(response.content)

import streamlit as st
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

# استيراد ملفاتك الخاصة
from Agent import build_agent
from Vector_db import vector_db
from chunking import chunking
from loader import load_documents_from_folder

# --- 1. إعدادات الصفحة ---
st.set_page_config(page_title="Seif AI Agent", page_icon="🤖")
st.title("🤖 RAG Multi-Tool Agent")

# --- 2. تهيئة الـ Vector DB والـ Agent (مرة واحدة فقط) ---
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
    
    # بناء الـ Agent
    agent_app = build_agent(vector, chunks)
    return agent_app

# تشغيل التهيئة
agent_app = initialize_system()

# --- 3. إدارة الذاكرة (Chat History) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# عرض الرسائل القديمة عند كل ريفريش
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. معالجة سؤال المستخدم ---
if prompt := st.chat_input("Ask me anything about your files or the web..."):
    
    # إضافة سؤال المستخدم للواجهة والذاكرة
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # تشغيل الـ Agent
    with st.chat_message("assistant"):
        message_placeholder = st.empty() # مكان للإجابة لحد ما تجهز
        message_placeholder.markdown("🔍 Thinking...")
        
        # إعداد الـ Config (نفس الـ Thread ID لضمان عمل الميموري)
        config = {"configurable": {"thread_id": "streamlit_session"}, "recursion_limit": 10}
        
        try:
            # مناداة الـ Agent
            input_data = {"messages": [HumanMessage(content=prompt)]}
            result = agent_app.invoke(input_data, config=config)
            
            # استخراج الإجابة النهائية
            final_answer = result["messages"][-1].content
            
            # عرض الإجابة
            message_placeholder.markdown(final_answer)
            
            # حفظ الإجابة في الذاكرة
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

# --- 5. Sidebar (اختياري) ---
with st.sidebar:
    st.info("Files loaded from '../data/' folder.")
    if st.button("Clear Chat Memory"):
        st.session_state.messages = []
        st.rerun()