import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from config import get_llm

def evaluate_response(query, context, response):

    parser = JsonOutputParser()
    judge_llm = get_llm()

    prompt = ChatPromptTemplate.from_template("""
    You are an expert Quality Assurance (QA) judge for Retrieval-Augmented Generation (RAG) systems.
    
    User Query: {query}
    Retrieved Context: {context}
    System Response: {response}

    Return STRICT JSON with:
    - score (1-10)
    - faithfulness_score (1-5)
    - relevance_score (1-5)
    - hallucination_detected (Boolean)
    - reasoning
    - improvement_suggestions
    """)

    chain = prompt | judge_llm | parser

    try:
        return chain.invoke({
            "query": query,
            "context": context,
            "response": response
        })
    except Exception as e:
        return {"error": f"Evaluation failed: {str(e)}"}