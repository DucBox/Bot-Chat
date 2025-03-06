import os
import xml.etree.ElementTree as ET
from src.utils import LOG_FILE_XML

chat_history = []

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
    Loads conversation history from the XML file and updates the global chat_history list.
    """
    global chat_history  

    if not os.path.exists(LOG_FILE_XML):
        print("⚠️ No previous chat history found.")
        return

    try:
        tree = ET.parse(LOG_FILE_XML)
        root = tree.getroot()
        chat_history.clear()  
        chat_history.extend([(conv.find("user").text, conv.find("bot").text) for conv in root.findall("conversation")])

        print(f"✅ Loaded {len(chat_history)} previous messages from history.")

    except Exception as e:
        print(f"⚠️ Error loading chat history: {str(e)}")
