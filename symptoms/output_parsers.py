from pydantic import BaseModel
from langchain_core.output_parsers import BaseOutputParser
import re

class DiseasePrediction(BaseModel):
    disease: str
    likelihood: int


class SymptomsOutput(BaseModel):
    predictions: list[DiseasePrediction]


class SymptomsOutputParser(BaseOutputParser[SymptomsOutput]):
    def parse(self, text: str) -> SymptomsOutput:
        print("start\n", text, "\nend")  # Debugging line to see raw output
        predictions = []

        pattern = r'"([^"]+)"\s*:\s*(\d+)%'
        matches = re.findall(pattern, text)

        for disease, likelihood in matches:
            predictions.append(DiseasePrediction(disease=disease.strip(), likelihood=int(likelihood)))

        return SymptomsOutput(predictions=predictions)

