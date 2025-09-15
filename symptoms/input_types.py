from pydantic import BaseModel

class SymptomsInput(BaseModel):
    symptoms: str
