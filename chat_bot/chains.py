from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough

from chat_bot.output_parsers import ChatBotOutputParser
from chat_bot.prompts import general_prompt, medical_prompt, router_prompt  # , symptoms_prompt

# from models import biomistral_llm
from models import gemma31b_llm, medgemma_llm

output_parser = ChatBotOutputParser()


def route_predicate(input_dict):
    print(input_dict)
    route = input_dict["route"].strip().lower()
    return route == "medical_expert"


def add_route(query):
    route = router_chain.invoke({"query": query})
    print(f"{query=}, {route=}")
    return {"query": query, "route": route}


def debug_parser(input_):
    print(f"start\n{input_}\nend")  # Debugging line to see raw output
    return input_


router_chain = router_prompt | gemma31b_llm | StrOutputParser()
general_chain = {"query": RunnablePassthrough(func=debug_parser)} | general_prompt | gemma31b_llm | output_parser
medical_chain = RunnablePassthrough(func=debug_parser) | medical_prompt | medgemma_llm | output_parser
chat_chain = RunnableLambda(func=add_route) | RunnableBranch((route_predicate, medical_chain), general_chain)
