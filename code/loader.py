import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader

def load_documents_from_folder(folder_path):
    documents = []
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} not found!")
        return []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
                print(f"Successfully loaded PDF: {filename}")

            elif filename.endswith(".pptx"):
                loader = UnstructuredPowerPointLoader(file_path)
                documents.extend(loader.load())
                print(f"Successfully loaded PPTX: {filename}")
        
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")

    return documents

