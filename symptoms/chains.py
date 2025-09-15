from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from models import biomistral_llm, gemma31b_llm, groq_llm
from symptoms.input_types import SymptomsInput
from symptoms.output_parsers import SymptomsOutputParser
from symptoms.prompts import json_converter_prompt, symptoms_prompt_cover, symptoms_prompt

output_parser = StrOutputParser()
output_parser = SymptomsOutputParser()

symptoms_chain = (
    RunnablePassthrough(input_type=SymptomsInput)
    | {
        "symptoms": lambda x: x["symptoms"],  # keep raw symptoms
        "llm_output": symptoms_prompt_cover | biomistral_llm,  # run few-shot + LLM
    }
    | json_converter_prompt
    | gemma31b_llm
)

groq_symptoms_chain = (
    RunnablePassthrough(input_type=SymptomsInput)
    | symptoms_prompt
    | groq_llm
)