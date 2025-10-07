from pydantic import BaseModel, Field
from typing import Optional


class ChatInput(BaseModel):
    """Input for the chat chain, optionally including image data."""

    query: str = Field(..., description="The user's text query.")
    image_url: Optional[str] = Field(None, description="Optional Base64 encoded image or URL.")


class LabReportInput(BaseModel):
    """Input for the lab report chain, optionally including image data."""

    images: list[str] = Field(default_factory=list, description="List of Base64 encoded image or URL.")


class MentslHealthInput(BaseModel):
    """Input for the chat chain, optionally including image data."""

    query: str = Field(..., description="The user's text query.")
