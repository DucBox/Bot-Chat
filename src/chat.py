import openai
import os

from src.config import OPENAI_API_KEY
from src.utils import count_tokens
from src.history import chat_history, save_history_to_xml
from src.summarizer import summarize_and_embed_chat_history
from src.files import embed_text
from src.retrieval import retrieve_relevant_chunks

# OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Token limit for history storage
TOKEN_LIMIT = 4000  
KEEP_LAST_N_PAIRS = 0  # Configurable: number of last pairs to keep

def chat_with_gpt(user_input):
    """
    Handles chatbot conversation, tracks tokens, retrieves document information, and manages history overflow.

    Args:
        user_input (str): The latest user message.

    Returns:
        str: The chatbot's response.
    """
    try:
        #Retrieve relevant document embeddings BEFORE responding
        retrieved_text = retrieve_relevant_chunks(user_input, top_k = 1)

        #Construct messages with FULL history + Retrieved Documents
        messages = [{"role": "system", "content": "You are an English teacher with 20 years of experience in IELTS training."}]

        for user_msg, bot_msg in chat_history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": bot_msg})

        if retrieved_text:
            messages.append({"role": "system", "content": f"Relevant document content retrieved:\n{retrieved_text}"})

        messages.append({"role": "user", "content": user_input})

        user_tokens = count_tokens(user_input)
        total_tokens = sum(count_tokens(m['content']) for m in messages)
        save_debug_prompt(user_input, messages, total_tokens)

        #Send request to OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )

        bot_response = response.choices[0].message.content
        bot_tokens = count_tokens(bot_response)
        total_tokens += bot_tokens
        

        print(f"📊 Token Count - User: {user_tokens} | Bot: {bot_tokens} | Total Prompt: {total_tokens}")

        chat_history.append((user_input, bot_response))

        #Handle token exceeds
        if total_tokens > TOKEN_LIMIT:
            print("⚠️ Token limit exceeded! Chunking and summarizing chat history...")
            summarize_and_embed_chat_history(chat_history)

            # Clear old history, keep last N pairs
            chat_history[:] = chat_history[-KEEP_LAST_N_PAIRS:] if KEEP_LAST_N_PAIRS > 0 else []

        save_history_to_xml()

        return bot_response
    except Exception as e:
        return f"⚠️ Error calling API: {str(e)}"
    

# # Debug directory for tracking prompts
# DEBUG_PROMPT_DIR = "debug_logs/prompts"
# os.makedirs(DEBUG_PROMPT_DIR, exist_ok=True)

# def save_debug_prompt(user_input, messages, total_tokens):
#     """
#     Saves each prompt sent to OpenAI, along with token count.

#     Args:
#         user_input (str): The user’s original input.
#         messages (list): The full conversation history sent to OpenAI.
#         total_tokens (int): Total token count of the request.
#     """
#     prompt_text = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])
    
#     debug_content = f"""============================
# 📝 **User Input:** {user_input}
# 📊 **Total Tokens:** {total_tokens}
# ---
# {prompt_text}
# ============================
# """

#     debug_file = os.path.join(DEBUG_PROMPT_DIR, "prompt_log.txt")

#     try:
#         with open(debug_file, "a", encoding="utf-8") as f:
#             f.write(debug_content + "\n\n")
#         print(f"✅ Debug: Prompt saved to {debug_file}")
#     except Exception as e:
#         print(f"⚠️ Error saving prompt debug file: {str(e)}")
