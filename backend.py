# main.py
# This is the main file that brings all the modules together.
# It simulates a modular project by defining the API routers within this single file.

import uvicorn
from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware
from api.chat_api import chat_router

# --- Main Application Logic ---
# This is where the main FastAPI app is initialized.

app = FastAPI(title="MedMinds - Backend APIs")


# The root endpoint from the original project.
@app.get("/")
def read_root():
    """
    Returns a simple welcome message as a JSON object.
    """
    return {"message": "Welcome to the Modular FastAPI server!"}


# The original hello endpoint.
@app.get("/hello/{name}")
def say_hello(name: str):
    """
    Greets a user by name.
    """
    return {"message": f"Hello, {name}!"}


# --- Include Routers ---
# Here, we 'include' the routers from our simulated modules into the main application.
# The `prefix` argument adds a common path prefix to all endpoints in the router.
# For example, the `/chat` endpoint in the router becomes `/api/v1/chat`.
app.include_router(chat_router, prefix="/api/v1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Main Execution Block ---
if __name__ == "__main__":
    # uvicorn.run starts the server.
    uvicorn.run("modular_project:app", host="127.0.0.1", port=8000, reload=True)
