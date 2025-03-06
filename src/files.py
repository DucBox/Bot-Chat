import os
import fitz  # PyMuPDF for PDF processing
import chromadb
import openai
from src.utils import BASE_DIR
from src.config import OPENAI_API_KEY

client = openai.OpenAI(api_key=OPENAI_API_KEY)

chroma_client = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "chroma_db"))
collection = chroma_client.get_or_create_collection(name="document_embeddings")

CHUNK_SIZE = 700  

DEBUG_EMBED_DIR = "debug_logs/embedded_doc"
os.makedirs(DEBUG_EMBED_DIR, exist_ok=True)

def save_debug_embedding(doc_id, text):
    """
    Saves a copy of embedded document chunks for debugging.

    Args:
        doc_id (str): The unique identifier for the document.
        text (str): The text content of the chunk.
    """
    debug_file = os.path.join(DEBUG_EMBED_DIR, f"embedded_{doc_id}.txt")

    try:
        with open(debug_file, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"✅ Debug: Embedded document chunk saved to {debug_file}")
    except Exception as e:
        print(f"⚠️ Error saving embedded document debug file: {str(e)}")

def chunk_text(text, chunk_size=CHUNK_SIZE):
    """
    Splits text into fixed-size chunks.

    Args:
        text (str): The full text to be chunked.
        chunk_size (int): The maximum number of words per chunk.

    Returns:
        list: A list of text chunks.
    """
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

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
    Splits text into chunks and embeds each chunk separately.

    Args:
        text (str): The text content to embed.
        doc_id (str): A unique identifier for the document.
    """
    chunks = chunk_text(text, chunk_size=CHUNK_SIZE)

    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk_{i}"
        try:
            response = client.embeddings.create(
                input=chunk,
                model="text-embedding-3-small"
            )
            embedding = response.data[0].embedding

            # Save debug copy before embedding
            save_debug_embedding(chunk_id, chunk)

            # Store in ChromaDB
            collection.add(
                documents=[chunk],
                metadatas=[{"doc_id": chunk_id}],
                embeddings=[embedding],
                ids=[chunk_id]
            )

            print(f"✅ Embedded chunk {i + 1}/{len(chunks)} for document: {doc_id}")

        except Exception as e:
            print(f"⚠️ Error embedding chunk {i}: {str(e)}")

def process_document(file_path):
    """
    Extracts text from a given file (PDF or TXT), chunks it, and embeds each chunk separately.

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
        doc_id = os.path.basename(file_path)  # Use filename as unique doc ID
        embed_text(text, doc_id)
        print(f"✅ Successfully chunked and embedded document: {file_path}")
