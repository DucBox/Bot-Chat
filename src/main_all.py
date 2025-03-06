import sys
import os
# Ensure the script can import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import openai
import xml.etree.ElementTree as ET
import tiktoken  # Token tracking
from src.config import OPENAI_API_KEY

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Paths for logging
LOG_DIR = "logs"
LOG_FILE_XML = os.path.join(LOG_DIR, "chat_history.xml")

# Ensure logs directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Global conversation history list
chat_history = []

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

def save_history_to_xml():
    """
    Saves the conversation history to an XML file.
    """
    root = ET.Element("chat_history")

    for user_msg, bot_msg in chat_history:
        conversation = ET.SubElement(root, "conversation")
        user_elem = ET.SubElement(conversation, "user")
        user_elem.text = user_msg
        bot_elem = ET.SubElement(conversation, "bot")
        bot_elem.text = bot_msg

    tree = ET.ElementTree(root)
    tree.write(LOG_FILE_XML, encoding="utf-8", xml_declaration=True)

def load_history_from_xml():
    """
    Loads conversation history from the XML file.
    """
    if not os.path.exists(LOG_FILE_XML):
        print("⚠️ No previous chat history found.")
        return

    try:
        tree = ET.parse(LOG_FILE_XML)
        root = tree.getroot()

        global chat_history
        chat_history = [(conv.find("user").text, conv.find("bot").text) for conv in root.findall("conversation")]

        print(f"✅ Loaded {len(chat_history)} previous messages from history.")

    except Exception as e:
        print(f"⚠️ Error loading chat history: {str(e)}")

def chat_with_gpt(user_input):
    """
    Sends a request to the chatbot, including the full conversation history.

    Args:
        user_input (str): The latest message from the user.

    Returns:
        str: The chatbot's response.
    """
    try:
        # Construct messages with full history
        messages = [{"role": "system", "content": "You are an English teacher with 20 years of experience in IELTS training."}]

        # Append entire conversation history
        for user_msg, bot_msg in chat_history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": bot_msg})

        # Add the latest user input
        messages.append({"role": "user", "content": user_input})

        # Convert messages to a formatted string for debugging token count
        full_prompt = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages])

        # Count tokens
        user_tokens = count_tokens(user_input)
        prompt_tokens = count_tokens(full_prompt)

        # Send request to OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )

        bot_response = response.choices[0].message.content

        # Count bot response tokens
        bot_tokens = count_tokens(bot_response)

        # Display token usage
        print(f"📊 Token Count - User: {user_tokens} | Bot: {bot_tokens} | Total Prompt: {prompt_tokens}")

        # Update history
        chat_history.append((user_input, bot_response))

        # Save history to XML
        save_history_to_xml()

        return bot_response
    except Exception as e:
        return f"⚠️ Error calling API: {str(e)}"

def chatbot_loop():
    """
    Starts the chatbot conversation loop.
    """
    print("\n🤖 Chatbot GPT-4o-mini (Type 'exit' to quit)")

    # Load history at startup
    load_history_from_xml()

    while True:
        user_input = input("\nUser: ")
        
        if user_input.lower() == "exit":
            print("👋 Goodbye! Your chat history has been saved.")
            break
        
        response = chat_with_gpt(user_input)
        print("\nChatbot:", response)

if __name__ == "__main__":
    chatbot_loop()
