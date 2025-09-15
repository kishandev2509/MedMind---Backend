from fastapi import APIRouter
from model.lc import get_response
from pydantic import BaseModel


class ChatRequest(BaseModel):
    input: str


# --- "chat_api.py" Module ---
# This section simulates a separate Python file named 'chat_api.py'.
# It contains a specific set of related endpoints.

# Initialize a new APIRouter. This acts like a mini-FastAPI application.
chat_router = APIRouter()


@chat_router.post("/chat")
def chat(request: ChatRequest):
    """
    Returns the response from the model for a given user query.
    """
    return get_response(request.input)
