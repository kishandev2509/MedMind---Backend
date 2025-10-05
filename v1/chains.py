from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from v1.prompts import chatbot_prompt, symptoms_prompt, lab_report_prompt
from v1.input_parsers import ChatInput, LabReportInput
from v1.models import medgemma_llm
from v1.utils import build_single_messages, lab_report_input_processor

chat_chain = (
    RunnablePassthrough.assign(prompt=chatbot_prompt)
    | RunnableLambda(build_single_messages).with_types(input_type=dict)
    | medgemma_llm
).with_types(input_type=ChatInput)

medgemma_symptoms_chain = (
    RunnablePassthrough.assign(prompt=symptoms_prompt)
    | RunnableLambda(build_single_messages).with_types(input_type=dict)
    | medgemma_llm
).with_types(input_type=ChatInput)


medgemma_lab_report_chain = (
    RunnablePassthrough.assign(prompt=lab_report_prompt)
    | RunnableLambda(lab_report_input_processor).with_types(input_type=dict)
    | medgemma_llm
).with_types(input_type=LabReportInput)
