from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory


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


store = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    """Gets the chat history for a given session ID."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def trim_history(inputs):
    """
    Trims the 'history' in the input dictionary to the last 10 messages.
    (5 user questions and 5 AI answers).
    """
    history = inputs.get("history", []) # Get history, default to empty list
    if len(history) > 20:
        # Keep only the most recent 10 messages
        inputs["history"] = history[-20:]
    return inputs