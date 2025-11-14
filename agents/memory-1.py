from mem0 import MemoryClient
from dotenv import load_dotenv

load_dotenv()

client = MemoryClient()

# messages = [
#    {
#        "role": "user",
#        "content": "Meu nome é Daniel e eu gosto de fazer automações com IA!",
#    },
#    {
#        "role": "assistant",
#        "content": "Oi Daniel! Anotei que você gosta de construir automações com IA! Vou manter isso em mente para recomendações e discussões relacionadas.",
#    },
# ]

# client.add(messages, user_id="daniel")

client.add("Sou o Daniel e gosto de robótica!", user_id="daniel")

query = "Qual o meu nome?"
response = client.search(query, filters={"user_id": "daniel"})
response
response["results"][0]["memory"]
