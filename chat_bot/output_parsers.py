import markdown
from langchain_core.output_parsers import BaseOutputParser
class ChatBotOutputParser(BaseOutputParser[str]):
    def parse(self, text: str) -> str:
        html = markdown.markdown(text)
        return html
