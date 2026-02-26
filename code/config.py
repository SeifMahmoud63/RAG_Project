from langchain_groq import ChatGroq
import os

def get_llm(temperature=0):
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=temperature
    )

