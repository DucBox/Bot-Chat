# 🤖 BOT ASSISTANT

## 📌 Introduction
BOT ASSISTANT is an AI-powered assistant that performs **summarization tasks** using OpenAI's GPT model. The project allows users to modify the summarization prompt and customize the assistant's behavior.

## 🚀 Features
✅ **Chatbot with Memory**: Remembers past interactions (until token limit exceeded).  
✅ **History Summarization & Embedding**: Compresses long conversations, stores them for retrieval.  
✅ **Document Upload & Retrieval**: Users can upload PDFs/TXTs, and the bot retrieves relevant content.  
✅ **Chunking**: Splits large documents into manageable parts for better search accuracy.  
✅ **Dynamic Prompt Modification**: Users can modify summarization prompts for better results.  

---

## 🛠 Installation

### 🔹 1. Clone the Project
```bash
git clone https://github.com/your-username/BOT-ASSISTANT.git
cd BOT-ASSISTANT
```

### 🔹 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🔥 Configuration
All configurations are located in `config.py`. Before running the project, set up your **OpenAI API key**.

### 🔹 1. Add OpenAI API Key in `config.py`
Modify `config.py` and add your **API key**:
```python
OPENAI_API_KEY = "your-openai-api-key"
```

### 🔹 2. Modify the Summarization Prompt
To customize the summarization behavior, edit the prompt in `summarizer.py`.

Example modification:
```python
PROMPT = "Summarize the following text in a concise and clear manner:"
```

---

## 🚀 Running the Application

### 🔹 1. Run the Assistant
```bash
python src/main.py
```

This will initialize the **BOT ASSISTANT** and allow you to test the summarization task.

---

## 🛠 Debugging & Troubleshooting

### 🔹 1. Verify API Key
Ensure that `OPENAI_API_KEY` is set correctly in `config.py`. You can test it with:
```python
from config import OPENAI_API_KEY
print(f"OpenAI Key: {OPENAI_API_KEY}")
```

### 🔹 2. Test Summarization Prompt
```python
from summarizer import summarize_text
text = "This is a long article that needs summarization."
summary = summarize_text(text)
print(summary)
```

---

## 📜 License & Author
- 📌 **Author:** Ngo Quang Duc
- 📌 **Contact:** quangducngo0811@gmail.com

🚀 Enjoy building with BOT ASSISTANT! 🎉

