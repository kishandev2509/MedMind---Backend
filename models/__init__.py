from langchain_groq import ChatGroq
from langchain_ollama import OllamaLLM


groq_llm = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0.5)
biomistral_llm = OllamaLLM(model="cniongolo/biomistral:latest")
gemma31b_llm = OllamaLLM(model="gemma3:1b")
medgemma_llm = OllamaLLM(model="alibayram/medgemma")
