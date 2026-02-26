import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from config import get_llm

class RAGJudge:
    def __init__(self):
        self.parser = JsonOutputParser()
        self.judge_llm=get_llm()
       
    def evaluate_response(self, query, context, response):
        prompt = ChatPromptTemplate.from_template("""
        You are an expert Quality Assurance (QA) judge for Retrieval-Augmented Generation (RAG) systems. 
        Your task is to evaluate the quality of the generated response based ONLY on the provided context.

        ### EVALUATION CRITERIA:
        1. **Faithfulness (Groundedness):** Is the answer derived solely from the retrieved context without adding outside info?
        2. **Answer Relevance:** Does the response directly and completely address the user's query?
        3. **Context Utilization:** How well did the agent use the provided chunks to formulate the answer?
        4. **Hallucination Check:** Does the answer contain any facts not present in the context?

        ### INPUT DATA:
        - **User Query:** {query}
        - **Retrieved Context:** {context}
        - **System Response:** {response}

        ### OUTPUT FORMAT:
        Return your evaluation in STRICT JSON format with the following keys:
        - "score": (Integer from 1 to 10)
        - "faithfulness_score": (1-5)
        - "relevance_score": (1-5)
        - "hallucination_detected": (Boolean)
        - "reasoning": (Short explanation of the scores)
        - "improvement_suggestions": (What could be better?)

        JSON Output:
        """)
        
        chain = prompt | self.judge_llm | self.parser
        
        try:
            evaluation = chain.invoke({
                "query": query,
                "context": context,
                "response": response
            })
            return evaluation
        except Exception as e:
            return {"error": f"Evaluation failed: {str(e)}"}