# Intelligent RAG Assistant
> An advanced agentic retrieval-augmented generation system for precise document intelligence.

---

## 01. System Overview
The **Intelligent RAG Assistant** is a sophisticated Question Answering framework designed to bridge the gap between static Large Language Models and dynamic, private data. By integrating a multi-stage retrieval pipeline and an autonomous decision layer, the system ensures high-fidelity responses grounded in verified document context.

---

## 02. Technical Architecture
The system follows a non-linear pipeline to transform raw queries into refined knowledge:

1.  **Input & Optimization**: Query Rewriting and HyDE (Hypothetical Document Embeddings) expansion.
2.  **Hybrid Retrieval**: Concurrent Vector Search and BM25 execution.
3.  **Refinement**: MMR (Maximal Marginal Relevance) for diversity and Flash Reranking for precision.
4.  **Agentic Layer**: Reasoning engine to determine retrieval necessity and domain alignment.
5.  **Memory Integration**: Persistence of conversation history for contextual continuity.

---

## 03. Core Capabilities

### Advanced Retrieval Engine
| Feature | Description |
| :--- | :--- |
| **Hybrid Search** | Merges semantic vector similarity with keyword-based BM25. |
| **MMR Ranking** | Minimizes redundancy by selecting diverse document chunks. |
| **Flash Reranker** | Applies cross-encoder scoring to prioritize the most relevant context. |

### Query Intelligence
* **Query Transformation**: Reformulates ambiguous user inputs into search-optimized queries.
* **HyDE Expansion**: Generates synthetic documents to improve embedding-based lookup in complex scenarios.

### Data Engineering
* **Recursive Splitting**: Intelligent chunking via `RecursiveCharacterTextSplitter`.
* **Token-Aware Encoding**: Utilizes `tiktoken` to maintain semantic integrity within model context windows.

---

## 04. The Agentic Layer
Beyond standard RAG, this system introduces an **Intelligent Decision Layer** that acts as a router for incoming requests:

* **Knowledge Routing**: Automatically determines if a query requires document retrieval or if it can be handled by the base LLM.
* **Domain Guardrails**: Filters out-of-scope questions to maintain system focus and reliability.
* **Stateful Memory**: Tracks the dialogue flow, allowing the assistant to resolve pronouns and follow-up questions seamlessly.

---

## 05. Technology Stack

| Component | Technology |
| :--- | :--- |
| **Core Language** | Python |
| **Orchestration** | LangChain |
| **Inference Engine** | Groq LLM APIs |
| **Vector Storage** | Chroma Database |
| **UI** | Streamlit |
| **Search/Rerank** | BM25, Flash Reranker |

---

## 06. Getting Started

### Environment Setup
The system requires the following environment variables:
* `GROQ_API_KEY`: For LLM inference.
* `CHROMA_DB_PATH`: Path for persistent vector storage.

### Installation
```bash
pip install -r requirements.txt
python main.py
