import sys
import os

# Ensure the script can import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.chat import chat_with_gpt
from src.history import load_history_from_xml
from src.files import process_document

def chatbot_loop():
    """
    Starts the chatbot conversation loop.
    """
    print("\n🤖 Chatbot GPT-4o-mini (Type 'exit' to quit)")

    # ✅ Load history BEFORE starting chatbot
    load_history_from_xml()

    while True:
        user_input = input("\nUser: ")
        
        if user_input.lower() == "upload":
            file_path = input("Enter the file path: ")
            process_document(file_path)
        elif user_input.lower() == "exit":
            print("👋 Goodbye! Your chat history has been saved.")
            break
        else:
            response = chat_with_gpt(user_input)
            print("\nChatbot:", response)

if __name__ == "__main__":
    chatbot_loop()
