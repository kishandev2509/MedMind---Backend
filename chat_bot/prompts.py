from langchain_core.prompts import ChatPromptTemplate

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

medical_prompt = ChatPromptTemplate.from_template("{query}")


