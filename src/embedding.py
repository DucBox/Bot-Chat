import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import os
import fitz 
import chromadb
import openai
from src.config import OPENAI_API_KEY

client = openai.OpenAI(api_key=OPENAI_API_KEY)

chroma_client = chromadb.PersistentClient(path="/Users/ngoquangduc/Desktop/AI_Project/chatbot_project/chroma_db")
collection = chroma_client.get_or_create_collection(name="document_embeddings")

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.
    
    Args:
        pdf_path (str): The path to the PDF file.
    
    Returns:
        str: The extracted text.
    """
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text() for page in doc)
        return text.strip()
    except Exception as e:
        print(f"⚠️ Error extracting text from PDF: {str(e)}")
        return ""

def extract_text_from_txt(txt_path):
    """
    Extracts text from a TXT file.
    
    Args:
        txt_path (str): The path to the TXT file.
    
    Returns:
        str: The extracted text.
    """
    try:
        with open(txt_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except Exception as e:
        print(f"⚠️ Error reading TXT file: {str(e)}")
        return ""

def embed_text(text, doc_id):
    """
    Generates embeddings for the extracted text and stores them in ChromaDB.

    Args:
        text (str): The text content to embed.
        doc_id (str): A unique identifier for the document.

    Returns:
        list: The embedding vector.
    """
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding
        
        # Store in ChromaDB
        collection.add(
            documents=[text],
            metadatas=[{"doc_id": doc_id}],
            embeddings=[embedding],
            ids=[doc_id]
        )

        return embedding
    except Exception as e:
        print(f"⚠️ Error generating embedding: {str(e)}")
        return []

def process_document(file_path):
    """
    Extracts text from a given file (PDF or TXT), generates embeddings, and stores them.

    Args:
        file_path (str): The path to the document.
    """
    if not os.path.exists(file_path):
        print("⚠️ File not found.")
        return
    
    ext = os.path.splitext(file_path)[-1].lower()
    
    if ext == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif ext == ".txt":
        text = extract_text_from_txt(file_path)
    else:
        print("⚠️ Unsupported file format. Only PDF and TXT are supported.")
        return
    
    if text:
        doc_id = os.path.basename(file_path) 
        embedding = embed_text(text, doc_id)
        if embedding:
            print(f"✅ Successfully stored embeddings for {file_path}")

if __name__ == "__main__":
    process_document("/Users/ngoquangduc/Desktop/AI_Project/chatbot_project/data/IELTS-Speaking-Part-1-Topics-Questions.pdf")
