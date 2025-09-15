from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnablePassthrough, RunnableLambda
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from pydantic import BaseModel
import re
from langchain_core.output_parsers import BaseOutputParser


class SymptomsInput(BaseModel):
    symptoms: str


class DiseasePrediction(BaseModel):
    disease: str
    likelihood: int


class SymptomsOutput(BaseModel):
    predictions: list[DiseasePrediction]


class SymptomsOutputParser(BaseOutputParser[SymptomsOutput]):
    def parse(self, text: str) -> SymptomsOutput:
        predictions = []

        # Regex to match: 1. Disease Name - 70%
        pattern = r"\d+\.\s*(.+?)\s*-\s*(\d+)%"
        matches = re.findall(pattern, text)

        for disease, likelihood in matches:
            predictions.append(DiseasePrediction(disease=disease.strip(), likelihood=int(likelihood)))

        return SymptomsOutput(predictions=predictions)


router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            'Given the user\'s question below, classify it as either "medical_expert" or "general_knowledge". Your classification should be based on the query\'s topic. If the question is about health, diseases, patients, medical procedures, medicine or any medical related topic then classify it as "medical_expert". For all other questions, classify it as "general_knowledge" and return the keyword only.',
        ),
        ("user", "{query}"),
    ]
)

general_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful and knowledgeable assistant. Answer the user's question or queries clearly and concisely."),
        ("user", "{query}"),
    ]
)

medical_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a highly specialized medical expert. Your role is to provide accurate and helpful information about health, diseases, patients, medical procedures, or medicine. if user query is not related to medical field, politely inform them that you can only answer medical related questions.",
        ),
        ("user", "{query}"),
    ]
)

# symptoms_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             """
#             Role: Medical Expert

#             Context: You are an AI language model designed to assist users by providing a list of possible diseases based on the symptoms they provide.

#             Task: Given a list of symptoms, your task is to analyze them and provide a list of three potential diseases that could be causing those symptoms.

#             Output Format:
#             1. Disease Name - Percentage%
#             2. Disease Name - Percentage%
#             3. Disease Name - Percentage%

#             Guidelines for Listing Diseases:
#             1. List atmost three diseases.
#             2. Each disease should be accompanied by a percentage indicating the likelihood of that disease being the cause of the symptoms.
#             3. The diseases should be sorted in descending order based on the percentage, with the most likely disease listed first.
#             4. The percentages should be realistic and based on common medical knowledge.
#             5. If the symptoms are vague or could be attributed to multiple conditions, use your best judgment to select the most relevant diseases.

#             Input Format:
#             symptom1, symptom2, symptom3, ...

#             Note:
#             - User will provide symptoms by seperating comma(,).
#             - Only provide the list of diseases with percentages as specified in the output format.
#             - Do not include any additional explanations, comments, or information outside of the specified output format.

#             Example Input:
#             fever, cough, sore throat

#             Example Output:
#             "predictions": [
#             {{"disease": "Influenza", "likelihood": 70}},
#             {{"disease": "Common Cold", "likelihood": 20}},
#             {{"disease": "COVID-19", "likelihood": 10}}
#             ]
#             """,
#         ),
#         ("user", "{symptoms}"),
#     ]
# )

symptoms_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a medical diagnosis assistant.
            Give exactly 3 possible diseases with likelihoods of in percentage based on the symptoms provided.
            You must ONLY reply in strict JSON following this schema:
            {{
                "symptoms": ["<list of symptoms>"],
                "possible_diseases": [
                    {{"disease": "<name>", "likelihood": "<High|Medium|Low>"}},
                    ...
                ]
            }}
            Do not include explanations, greetings, or extra text. Output must be valid JSON only.
            """,
        ),
        ("human", "{symptoms}"),
    ]
)


groq_llm = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0.5)
biomistral_llm = OllamaLLM(model="cniongolo/biomistral:latest")

output_parser = StrOutputParser()


def route_predicate(input_dict):
    print(input_dict)
    route = input_dict["route"].strip().lower()
    return route == "medical_expert"


def add_route(query):
    route = router_chain.invoke({"query": query})
    print(f"{query=}, {route=}")
    return {"query": query, "route": route}


def add_symptoms_prefix(input: dict):
    print("start", input, "end")
    return {"symptoms": input["symptoms"]}


symptoms_output_parser = SymptomsOutputParser()

router_chain = router_prompt | groq_llm | output_parser
symptoms_chain = RunnablePassthrough(input_type=SymptomsInput) | symptoms_prompt | biomistral_llm | output_parser
general_chain = {"query": RunnablePassthrough(func=lambda x: x["root"])} | general_prompt | groq_llm | output_parser
medical_chain = {"query": RunnablePassthrough()} | medical_prompt | biomistral_llm | output_parser
chain = RunnableLambda(func=add_route) | RunnableBranch((route_predicate, medical_chain), general_chain)
