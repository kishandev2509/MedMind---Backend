from langchain_core.output_parsers import BaseOutputParser
from pydantic import BaseModel, Field


class ChatBotOutputParser(BaseOutputParser[str]):
    def parse(self, text: str) -> str:
        return text


class PossibleDisease(BaseModel):
    """Schema for a single disease and its likelihood."""
    disease: str = Field(description="The name of a potential disease.")
    likelihood: int = Field(description="The likelihood percentage (0-100) that this disease is correct.")

class SymptomOutput(BaseModel):
    """The complete structured output schema for symptom analysis."""
    symptoms: str = Field(description="A comma-separated string listing all symptoms identified from the user's query.")
    possible_disease: list[PossibleDisease] = Field(description="A list of possible diseases matching the symptoms, with likelihoods summing to 100.")
