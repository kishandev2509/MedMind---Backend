import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

from chat_bot.chains import chat_chain
from symptoms.chains import groq_symptoms_chain, symptoms_chain

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(title="MedMinds - Backend APIs", version="1.0", description="APIs for MedMinds - Your AI-Powered Medical Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def is_ollama_running() -> bool:
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False
    
if is_ollama_running():
    # Add LangServe routes
    add_routes(app, chat_chain, path="/chat")
    add_routes(app, symptoms_chain, path="/symptoms")
    add_routes(app, groq_symptoms_chain, path="/symptoms/v2")
else:
    print("❌ Ollama is not running. Start it with `ollama serve`.")
    print("⚠️ Skipping Ollama routes, service not available.")

if __name__ == "__main__":
    print("🚀 Starting MedMinds backend...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
