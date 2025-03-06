import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
from src.files import process_document
from src.chat import chat_with_gpt
from src.history import chat_history

st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")

st.title("🤖 AI Assistant with RAG")

st.sidebar.header("📂 Upload Documents")
uploaded_file = st.sidebar.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file:
    file_path = os.path.join("temp_files", uploaded_file.name)
    os.makedirs("temp_files", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.sidebar.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")
    process_document(file_path)  
    st.sidebar.info("🔄 Document processed and embedded!")

st.header("💬 Chat with AI")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for user_msg, bot_msg in st.session_state.chat_history:
    st.chat_message("user").write(user_msg)
    st.chat_message("assistant").write(bot_msg)

user_input = st.chat_input("Ask me anything...")

if user_input:
    response = chat_with_gpt(user_input)
    
    st.session_state.chat_history.append((user_input, response))
    
    st.chat_message("user").write(user_input)
    st.chat_message("assistant").write(response)
