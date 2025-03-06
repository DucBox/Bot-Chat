import os
import openai
import datetime
from src.config import OPENAI_API_KEY
from src.files import embed_text

client = openai.OpenAI(api_key=OPENAI_API_KEY)

CHUNK_SIZE = 700  

# Debug directories
DEBUG_CHUNK_DIR = "debug_logs/chunks"
DEBUG_SUMMARY_DIR = "debug_logs/summaries"
os.makedirs(DEBUG_CHUNK_DIR, exist_ok=True)
os.makedirs(DEBUG_SUMMARY_DIR, exist_ok=True)

def save_debug_file(directory, filename, content):
    """
    Saves debugging files for chunking and summarization.

    Args:
        directory (str): The directory to save the file in.
        filename (str): The filename.
        content (str): The text content to save.
    """
    file_path = os.path.join(directory, filename)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ Debug: Saved {filename} to {directory}")
    except Exception as e:
        print(f"⚠️ Error saving debug file {filename}: {str(e)}")

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

def summarize_chunk(chunk, chunk_id):
    """
    Summarizes a single chat history chunk using OpenAI with a structured prompt.

    Args:
        chunk (str): The text chunk to summarize.
        chunk_id (int): Chunk ID for debugging.

    Returns:
        str: A structured summary of the chunk.
    """
    prompt = f"""
    You are an AI assistant summarizing a structured conversation between a **user and a chatbot assistant** in the field of **education and AI**.

    **Your Goal:**
    - Summarize the conversation **without missing any important details**.
    - Ensure that **specific information like names, dates, times, locations, numbers, and key terms are always included**.
    - Organize the information in a **graph-style format** to ensure clarity.

    **Formatting Rules:**
    - Maintain a structured format.
    - Do NOT remove key details (dates, times, locations, numbers, names).
    - Keep technical information precise.

    📝 **Final Notes & Decisions**:
    - [Action items or conclusions]
    ```
    **Conversation Chunk to Summarize:**
    {chunk}

    **Provide a structured summary based on the above format.**
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are an expert AI summarizer."},
                      {"role": "user", "content": prompt}]
        )

        summary = response.choices[0].message.content.strip()

        # ✅ Save debug logs
        save_debug_file(DEBUG_SUMMARY_DIR, f"summary_chunk_{chunk_id}.txt", summary)

        return summary

    except Exception as e:
        return f"⚠️ Error generating summary: {str(e)}"

def summarize_and_embed_chat_history(chat_history):
    """
    Summarizes the full chat history using chunking, embeds the summaries, and clears old history.

    Args:
        chat_history (list): List of (user, bot) conversation pairs.
    """
    full_chat_text = "\n".join([f"User: {user}\nBot: {bot}" for user, bot in chat_history])

    chunks = chunk_text(full_chat_text, chunk_size=CHUNK_SIZE)

    for i, chunk in enumerate(chunks):
        save_debug_file(DEBUG_CHUNK_DIR, f"chunk_{i}.txt", chunk)

    summarized_chunks = [summarize_chunk(chunk, i) for i, chunk in enumerate(chunks)]

    for i, summary in enumerate(summarized_chunks):
        embed_text(summary, f"chat_summary_chunk_{i}")

    print(f"✅ Successfully summarized and embedded {len(summarized_chunks)} chat chunks.")
