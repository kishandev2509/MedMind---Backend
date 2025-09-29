from langchain_core.runnables import RunnableLambda

from v1.input_parsers import ChatInput
from v1.output_parsers import ChatBotOutputParser
from v1.models import medgemma_llm, structured_medgemma_llm
from v1.utils import build_single_messages
from v1.prompts import symptoms_prompt

chatbot_output_parser = ChatBotOutputParser()

chat_chain = RunnableLambda(build_single_messages).with_types(input_type=ChatInput) | medgemma_llm | chatbot_output_parser
medgemma_symptoms_chain = symptoms_prompt | structured_medgemma_llm
