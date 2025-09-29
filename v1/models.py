from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from v1.output_parsers import SymptomAnalysis

medgemma_llm = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0.5)
biomistral_llm = ChatOllama(model="cniongolo/biomistral:latest")
gemma31b_llm = ChatOllama(model="gemma3:1b")
gemma34b_llm = ChatOllama(model="gemma3:4b")
medgemma_llm = ChatOllama(model="amsaravi/medgemma-4b-it:q6")
structured_medgemma_llm = medgemma_llm.with_structured_output(SymptomAnalysis)
