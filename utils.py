from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pypdf import PdfReader

# Retrieval of file data
def extract_pdf(file) :
    reader = PdfReader(file)
    text = ""   # this is a string type varivale
    for page in reader.pages :
        text += page.extract_text()
    return text
        
# splitting 

def split_text(text) :
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)

    return chunks   

# Embeddings and vectore storage

def create_vector_text(text) :
    chunks = split_text(text)
    docs = [Document(page_content=c) for c in chunks]
    embedding = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
    vectorestore = FAISS.from_documents(docs,embedding)
    
    return vectorestore
    


