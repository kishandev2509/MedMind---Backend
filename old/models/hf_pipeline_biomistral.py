from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = "BioMistral/BioMistral-7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
print("tokenizer initiated")
model = AutoModelForCausalLM.from_pretrained(model_id)
print("model initiated")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)
print("pipeline initiated")
biomistral_llm = HuggingFacePipeline(pipeline=pipe)
print("hf biomistral llm initiated")
