import subprocess
import httpx
import uvicorn
import asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

from v1.chains import chat_chain_with_memory, medgemma_symptoms_chain, medgemma_lab_report_chain, mental_health_chain_with_memory

load_dotenv()


def add_routes_to_app():
    """Adds routes to the FastAPI app."""
    add_routes(app, chat_chain_with_memory, path="/chat")
    add_routes(app, medgemma_symptoms_chain, path="/symptom_checker")
    add_routes(app, medgemma_lab_report_chain, path="/lab_report_analysis")
    add_routes(app, mental_health_chain_with_memory, path="/mental_health_support")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """This function runs on application startup and shutdown. It will check for and attempt to start the Ollama server."""
    print("🚀 Application startup...")

    async def is_ollama_running_async() -> bool:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:11434/api/tags", timeout=2.0)
                return response.status_code == 200
        except Exception:
            return False

    if await is_ollama_running_async():
        print("✅ Ollama is already running. Adding routes.")
        add_routes_to_app()
    else:
        print("⚠️ Ollama is not running. Attempting to start it in the background...")
        try:
            subprocess.Popen(["ollama", "serve"])
            print("Waiting 5 seconds for Ollama to start...")
            await asyncio.sleep(5)  # Use async sleep

            if await is_ollama_running_async():
                print("✅ Ollama started successfully. Adding routes.")
                add_routes_to_app()
            else:
                print("❌ Failed to start Ollama. Skipping Ollama routes.")

        except FileNotFoundError:
            print("❌ 'ollama' command not found. Please ensure Ollama is installed and in your system's PATH.")
        except Exception as e:
            print(f"❌ An unexpected error occurred while trying to start Ollama: {e}")

    # The 'yield' is where your application will run
    yield

    # Code after the 'yield' runs on shutdown
    print("🔌 Application shutdown...")


app = FastAPI(
    title="MedMinds - Backend APIs",
    version="1.0",
    description="APIs for MedMinds - Your AI-Powered Medical Assistant",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    print("🚀 Starting MedMinds backend with Uvicorn...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
