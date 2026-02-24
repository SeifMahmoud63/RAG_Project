Intelligent RAG Assistant
Overview

This project is an intelligent Question Answering system built using Retrieval Augmented Generation (RAG). The main idea behind this project is to improve answer accuracy by combining large language models with retrieval-based search over custom documents rather than relying only on general model knowledge.

The system is designed to behave like a smart knowledge assistant that can understand user questions, search for the most relevant information inside documents, and generate clear and natural responses.

Features
Advanced Retrieval Techniques

Hybrid Search combining Vector Search and BM25 for better retrieval accuracy.

Maximal Marginal Relevance (MMR) to reduce redundancy and increase result diversity.

Re-ranking using Flash Reranker to improve final result quality.

Query Understanding

Query Rewriting to improve search performance by reformulating user questions.

HyDE (Hypothetical Document Embeddings) to enhance understanding of complex or unclear queries.

Data Processing

Multi-PDF document support.

Recursive text splitting using RecursiveCharacterTextSplitter.

Token-aware splitting using tiktoken encoder to preserve semantic meaning.

Agent Intelligence and Memory

The project introduces an intelligent decision layer beyond traditional RAG systems.

Agent Layer

The agent helps decide how to respond to user queries by determining:

Whether the question requires document retrieval.

Whether the question can be answered directly by the language model.

Whether the question is outside the knowledge domain.

This makes the system more adaptive and closer to real-world intelligent assistants.

Memory System

The memory component allows the system to maintain conversation context. It helps:

Remember previous conversation history.

Maintain dialogue continuity.

Provide more personalized and coherent responses.

Architecture

The pipeline is structured as follows:

User Query

Query Optimization and Rewriting

HyDE Expansion

Hybrid Retrieval (Vector + BM25)

Diversity Ranking using MMR

Re-ranking using Flash Reranker

Response Generation

Agent Decision Layer

Memory Integration

Technologies

Python

LangChain

Groq LLM APIs

Chroma Vector Database

FastAPI

Flash Reranker

BM25 Retrieval
