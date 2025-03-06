import openai
import os

from src.config import OPENAI_API_KEY
from src.utils import count_tokens
from src.history import chat_history, save_history_to_xml
from src.summarizer import summarize_and_embed_chat_history
from src.files import embed_text
from src.retrieval import retrieve_relevant_chunks

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Token limit for history storage
TOKEN_LIMIT = 4000  
KEEP_LAST_N_PAIRS = 0  # Keep last 3 pairs before resetting history

def chat_with_gpt(user_input):
    """
    Handles chatbot conversation, tracks tokens, retrieves document information, and manages history overflow.

    Args:
        user_input (str): The latest user message.

    Returns:
        str: The chatbot's response.
    """
    # try:
    # Retrieve relevant document embeddings BEFORE responding
    retrieved_texts = retrieve_relevant_chunks(user_input, top_k=3)
    
    # Build <History Chat> section
    history_chat = "\n".join([f"User: {user}\nAssistant: {bot}" for user, bot in chat_history])
    history_chat = f"<History Chat>\n{history_chat}\n</History Chat>" if history_chat else "<History Chat>\n(No prior chat history)\n</History Chat>"
    
    # Build <Memory> section
    memory_section = "<Memory>\n"
    user_uploaded_data = []
    summarized_chat_history = []
    
    for text in retrieved_texts:
        if "This is a summarization of a part of history chat" in text:
            summarized_chat_history.append(text)
        else:
            user_uploaded_data.append(text)
    
    if user_uploaded_data:
        memory_section += "[User-Uploaded Documents]\n" + "\n".join([f"- {doc}" for doc in user_uploaded_data]) + "\n"
    
    if summarized_chat_history:
        memory_section += "[Past Summarized History]\n" + "\n".join([f"- {summary}" for summary in summarized_chat_history]) + "\n"
    
    memory_section += "</Memory>"
    
    # Construct background instructions for LLM
    background_section = (
        "Background: You are an assistant with 20 years of experience. "
        "The **History Chat** section contains the most recent user interactions and bot responses. "
        "The **Memory** section consists of two types of documents: "
        "1️⃣ **User-uploaded files** (which contain facts, reports, or guidelines). "
        "2️⃣ **Summarized past chat history** (which condenses older conversations for long-term memory). "
        "\n\nImportant Rules: "
        "- If recent history **conflicts** with past summarized history, prioritize the most **recent conversation**. "
        "- If answering from **uploaded documents**, state **'[Source: User-Uploaded Document]'**. "
        "- If answering from **summarized history**, state **'[Source: Past History Chat]'**. "
        "- If no relevant information is found, politely indicate that the bot **does not have the required data** and then response I DON'T KNOW, do not make up the answer in any case."
    )
    
    # Final user input section
    user_prompt = f"Current User Input: {user_input}"
    
    # Final formatted prompt
    final_prompt = f"""
    {history_chat}\n\n{memory_section}\n\n{background_section}\n\n{user_prompt}
    """.strip()
    
    # Send request to OpenAI
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": final_prompt}]
    )
    
    bot_response = response.choices[0].message.content.strip()
    
    # Track tokens
    total_tokens = count_tokens(final_prompt) + count_tokens(bot_response)
    print(f"📊 Token Count - Total Prompt: {total_tokens}")
    
    # Append to history
    chat_history.append((user_input, bot_response))
    save_debug_prompt(user_input, final_prompt, total_tokens)
    # Handle token overflow
    if total_tokens > TOKEN_LIMIT:
        print("⚠️ Token limit exceeded! Chunking and summarizing chat history...")
        summarize_and_embed_chat_history(chat_history)
        chat_history[:] = chat_history[-KEEP_LAST_N_PAIRS:] if KEEP_LAST_N_PAIRS > 0 else []
    
    save_history_to_xml()
    
    return bot_response
    # except Exception as e:
    #     return f"⚠️ Error calling API: {str(e)}"

# Debug directory for tracking prompts
DEBUG_PROMPT_DIR = "debug_logs/prompts"
os.makedirs(DEBUG_PROMPT_DIR, exist_ok=True)

def save_debug_prompt(user_input, messages, total_tokens):
    """
    Saves each prompt sent to OpenAI, along with token count.

    Args:
        user_input (str): The user’s original input.
        messages (list): The full conversation history sent to OpenAI.
        total_tokens (int): Total token count of the request.
    """
    prompt_text = messages
    
    debug_content = f"""============================
    📝 **User Input:** {user_input}
    📊 **Total Tokens:** {total_tokens}
    ---
    {prompt_text}
    ============================
    """

    debug_file = os.path.join(DEBUG_PROMPT_DIR, "prompt_log.txt")

    try:
        with open(debug_file, "a", encoding="utf-8") as f:
            f.write(debug_content + "\n\n")
        print(f"✅ Debug: Prompt saved to {debug_file}")
    except Exception as e:
        print(f"⚠️ Error saving prompt debug file: {str(e)}")
