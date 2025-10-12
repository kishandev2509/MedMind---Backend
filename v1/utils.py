from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage


def create_multimodal_human_message(input_dict: dict) -> HumanMessage:
    """Creates the HumanMessage content, handling both text and image."""
    content = [{"type": "text", "text": input_dict["query"]}]
    if input_dict.get("image_url"):
        image_content_block = {
            "type": "image_url",
            "image_url": {"url": input_dict["image_url"]},
        }
        content.append(image_content_block)
    return HumanMessage(content=content)


def build_single_messages(input_dict: dict):
    """Combines the new, comprehensive System Prompt and the Multimodal Human Message."""
    system_text = input_dict["prompt"].messages[0].content
    system_message = SystemMessage(content=system_text)
    human_message = create_multimodal_human_message(input_dict)
    return [system_message, human_message]


def lab_report_input_processor(input_dict: dict) -> list:
    """Creates a single HumanMessage with multiple content parts for a multimodal model."""
    system_text = input_dict["prompt"].messages[0].content
    content = []
    if input_dict.get("images"):
        for image_file in input_dict.get("images", []):
            content.append({"type": "image_url", "image_url": {"url": image_file}})
    system_message = SystemMessage(content=system_text)
    human_message = HumanMessage(content=content)
    return [system_message, human_message]


mental_health_store = {}
chat_store = {}


def get_chat_session_history(user_id: str) -> ChatMessageHistory:
    """Gets the chat history for a given session ID. in mental health"""
    if user_id not in chat_store:
        chat_store[user_id] = ChatMessageHistory()
    return chat_store[user_id]

def get_mh_session_history(user_id: str) -> ChatMessageHistory:
    """Gets the chat history for a given session ID. in mental health"""
    if user_id not in mental_health_store:
        mental_health_store[user_id] = ChatMessageHistory()
    return mental_health_store[user_id]


def trim_history(inputs):
    """
    Trims the 'history' in the input dictionary to the last 10 messages.
    (5 user questions and 5 AI answers).
    """
    mem_len = 5
    history = inputs.get("history", [])  # Get history, default to empty list
    if len(history) > mem_len:
        # Keep only the most recent 10 messages
        inputs["history"] = history[-mem_len:]
    return inputs
