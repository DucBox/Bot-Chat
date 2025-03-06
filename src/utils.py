import os
import tiktoken  # Token tracking

BASE_DIR = "/Users/ngoquangduc/Desktop/AI_Project/chatbot_project/chroma_db"
LOG_DIR = "/Users/ngoquangduc/Desktop/AI_Project/chatbot_project/src/logs"
LOG_FILE_XML = os.path.join(LOG_DIR, "chat_history.xml")

# Ensure logs directory exists
os.makedirs(LOG_DIR, exist_ok=True)

def count_tokens(text):
    """
    Counts the number of tokens in a given text using tiktoken.

    Args:
        text (str): The text to be tokenized.

    Returns:
        int: The token count.
    """
    encoding = tiktoken.encoding_for_model("gpt-4o")
    return len(encoding.encode(text))
