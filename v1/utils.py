from langchain_core.messages import HumanMessage, SystemMessage
from v1.prompts import chatbot_prompt

def create_multimodal_human_message(input_dict: dict) -> HumanMessage:
    """Creates the HumanMessage content, handling both text and image."""
    content = [{"type": "text", "text": input_dict["query"]}]

    # Using 'image_url' (your Pydantic key) to access the Base64 string
    if input_dict.get("image_url"):
        image_content_block = {
            "type": "image_url",
            "image_url": {"url": input_dict["image_url"]},
        }
        content.append(image_content_block)

    return HumanMessage(content=content)


def build_single_messages(input_dict: dict):
    """Combines the new, comprehensive System Prompt and the Multimodal Human Message."""
    # Access the system prompt content from the new single_model_system_prompt
    system_text = chatbot_prompt.messages[0].prompt.template
    system_message = SystemMessage(content=system_text)

    human_message = create_multimodal_human_message(input_dict)
    return [system_message, human_message]

