from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

client = OpenAI(base_url="https://api.groq.com/openai/v1")


class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


response = client.responses.parse(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    input="Daniel e Alberto vão gravar uma aula na terça-feira.",
    instructions="Extraia informações do evento.",
    text_format=CalendarEvent,
)

event = response.output_parsed
event.date
event.name
event.participants

print(event.model_dump_json(indent=2))
