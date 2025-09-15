from langchain_core.prompts import ChatPromptTemplate, FewShotPromptTemplate, PromptTemplate

# symptoms_prompt = ChatPromptTemplate.from_template(
#     """
#     Role: Medical Expert

#     Context: You are an AI language model designed to assist users by providing a list of possible diseases based on the symptoms they provide.

#     Task: Given a list of symptoms, your task is to analyze them and provide a list of three potential diseases that could be causing those symptoms.

#     Input Format:
#     symptom1, symptom2, symptom3, ...

#     Output Format:
#     You must ONLY reply in strict JSON following this schema:
#     {{
#         "symptoms": ["<list of symptoms>"],
#         "possible_diseases": [
#             {{"disease": "<name>", "likelihood": "<integer percentage>"}},
#             ...
#         ]
#     }}

#     Example Input:
#     fever, cough, sore throat

#     Example Output:
#     {{
#     "symptoms": ["fever", "cough", "sore throat"],
#     "possible_diseases": [
#         {{"disease": "Influenza", "likelihood": 70}},
#         {{"disease": "Common Cold", "likelihood": 20}},
#         {{"disease": "COVID-19", "likelihood": 10}}
#     ]
#     }}

#     Guidelines for Listing Diseases:
#     1. List exactly three diseases.
#     2. Each disease should be accompanied by a percentage indicating the likelihood of that disease being the cause of the symptoms like shown in output format.
#     3. The diseases should be sorted in descending order based on the percentage, with the most likely disease listed first.
#     4. The percentages should be realistic and based on common medical knowledge.
#     5. If the symptoms are vague or could be attributed to multiple conditions, use your best judgment to select the most relevant diseases.

#     Note:
#     - User will provide symptoms by seperating comma(,).
#     - Only provide the list of diseases with percentages as specified in the output format.
#     - Do not include any additional explanations, comments, or information outside of the specified output format.

#     {symptoms}
#     """
# )

# symptoms_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             """
#             You are a medical diagnosis assistant.
#             Give exactly 3 possible diseases with likelihoods of in percentage based on the symptoms provided.
#             You must ONLY reply in strict JSON following this schema:
#             {{
#                 "symptoms": ["<list of symptoms>"],
#                 "possible_diseases": [
#                     {{"disease": "<name>", "likelihood": "<High|Medium|Low>"}},
#                     ...
#                 ]
#             }}
#             Do not include explanations, greetings, or extra text. Output must be valid JSON only.
#             """,
#         ),
#         ("human", "{symptoms}"),
#     ]
# )


# Function to format list of dicts into a string
def format_diseases(disease_list):
    return ", ".join([f"({d['disease']}: {d['likelihood']}%)" for d in disease_list])


example_template = PromptTemplate(
    input_variables=["symptoms", "possible_disease"],
    template="Symptoms: {symptoms}\nPossible Disease: {possible_disease}",
)

examples = [
    {
        "symptoms": "[fever, cough, sore throat]",
        "possible_disease": format_diseases(
            [
                {"disease": "Influenza", "likelihood": 70},
                {"disease": "Common Cold", "likelihood": 20},
                {"disease": "COVID-19", "likelihood": 10},
            ]
        ),
    },
    {
        "symptoms": "[cough]",
        "possible_disease": format_diseases(
            [
                {"disease": "Bronchitis", "likelihood": 80},
                {"disease": "Pneumonia", "likelihood": 15},
                {"disease": "Asthma", "likelihood": 5},
            ]
        ),
    },
]

prefix = "Here are some examples:"
suffix = "Now, tell me the possible disease for [{symptoms}] and strictly return json only."

symptoms_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    prefix=prefix,
    suffix=suffix,
    input_variables=["symptoms"],
)

symptoms_prompt_cover = ChatPromptTemplate.from_messages([("human", symptoms_prompt.format(symptoms="{symptoms}"))])


json_converter_prompt = ChatPromptTemplate.from_template(
    """
    Format the disease and its likelihood into strict JSON following this schema:
    {{
        "symptoms": ["{symptoms}"],
        "possible_diseases": {{
            "<disease>": "<integer>",
            ...
        }}
    }}

    Ensure the output is valid JSON only, without any additional text or explanations.
    Text to convert: {llm_output}
    """
)

if __name__ == "__main__":
    print(symptoms_prompt.format(symptoms="headache, nausea, sensitivity to light"))
