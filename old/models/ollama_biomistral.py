# --- IMPORTANT SETUP FOR BIOMISTRAL-7B ---
# This example assumes you are running BioMistral-7B locally using a service like Ollama.
# 1. Download and install Ollama from https://ollama.com.
# 2. Pull the BioMistral-7B model by running: `ollama pull biomistral:7b`
# 3. This script will then connect to the local Ollama instance.
# This is the placeholder for BioMistral-7B. We use Ollama to run a local model.
# Make sure Ollama is running and you have pulled the model.
from langchain_ollama import OllamaLLM
biomistral_llm = OllamaLLM(model="cniongolo/biomistral:latest")
print("ollama biomistral llm created")