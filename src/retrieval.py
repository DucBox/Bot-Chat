import chromadb
import os
import openai 
from src.utils import BASE_DIR
from src.config import OPENAI_API_KEY

client = openai.OpenAI(api_key=OPENAI_API_KEY)

chroma_client = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "chroma_db"))

try:
    collection = chroma_client.get_collection(name="document_embeddings")
except chromadb.errors.InvalidCollectionException:
    print("⚠️ Collection 'document_embeddings' not found. Creating a new one...")
    collection = chroma_client.create_collection(name="document_embeddings")

def retrieve_relevant_chunks(query, top_k=3):
    """
    Retrieves the most relevant document chunks from ChromaDB based on the user's query.

    Args:
        query (str): The user's question.
        top_k (int): Number of top relevant chunks to retrieve.

    Returns:
        list: A list of retrieved text chunks.
    """
    try:
        response = client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding

        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

        if results["documents"]:
            return results["documents"][:top_k]  # ✅ Retrieve multiple relevant chunks
        else:
            return ["I couldn't find relevant information in the uploaded documents."]

    except Exception as e:
        return [f"⚠️ Error retrieving information: {str(e)}"]
