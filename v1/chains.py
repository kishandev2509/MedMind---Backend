from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

from v1.input_parsers import ChatInput, LabReportInput, MentslHealthInput
from v1.models import medgemma_llm
from v1.prompts import chatbot_prompt, lab_report_prompt, mental_health_prompt, symptoms_prompt
from v1.utils import build_single_messages, get_chat_session_history, get_mh_session_history, lab_report_input_processor, trim_history

chat_core_chain = RunnableLambda(trim_history) | chatbot_prompt | medgemma_llm

chat_chain_with_memory = RunnableWithMessageHistory(
    chat_core_chain,
    get_chat_session_history,
    input_messages_key="query",  # The key for the user's input in the invoke() call
    history_messages_key="history",  # The key for the history in the prompt
).with_types(input_type=ChatInput)

chat_chain = (
    RunnablePassthrough.assign(prompt=chatbot_prompt) | RunnableLambda(build_single_messages).with_types(input_type=dict) | medgemma_llm
).with_types(input_type=ChatInput)

medgemma_symptoms_chain = (
    RunnablePassthrough.assign(prompt=symptoms_prompt) | RunnableLambda(build_single_messages).with_types(input_type=dict) | medgemma_llm
).with_types(input_type=ChatInput)


medgemma_lab_report_chain = (
    RunnablePassthrough.assign(prompt=lab_report_prompt) | RunnableLambda(lab_report_input_processor).with_types(input_type=dict) | medgemma_llm
).with_types(input_type=LabReportInput)


mental_health_core_chain = RunnableLambda(trim_history) | mental_health_prompt | medgemma_llm

mental_health_chain_with_memory = RunnableWithMessageHistory(
    mental_health_core_chain,
    get_mh_session_history,
    input_messages_key="query",  # The key for the user's input in the invoke() call
    history_messages_key="history",  # The key for the history in the prompt
).with_types(input_type=MentslHealthInput)


# --- How to use the new chain ---
# You now need to provide a session_id in the config for each call

# First message from a user
# response = chain_with_memory.invoke(
#     {"query": "I'm feeling really anxious today."},
#     config={"configurable": {"session_id": "user123"}}
# )
# print(response)

# Second message from the same user
# The chain will now remember the first message.
# response_2 = chain_with_memory.invoke(
#     {"query": "What's a simple breathing exercise I can do?"},
#     config={"configurable": {"session_id": "user123"}}
# )
# print(response_2)
