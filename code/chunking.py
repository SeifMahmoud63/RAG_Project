from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunking(docs):
    splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250,chunk_overlap=50)

    return splitter.split_documents(docs)
     