import os
import uuid

from dotenv import load_dotenv
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

""" PIPELINE COMPLETO (básico):
1. Configurações e inicialização do banco de vetores;
2. Ingestão e pré-processamento dos dados;
3. Indexação dos dados no banco de vetores;
4. Consulta e recuperação de informações relevantes.
"""

""" CONFIGURAÇÕES E INICIALIZAÇÃO DO BANCO DE VETORES:
- definição de constantes,
- criação do cliente do Qdrant, e
- criação da coleção no Qdrant (caso ainda não exista)."""

"""
Primeiramente é preciso criar um cluster no Qdrant (https://cloud.qdrant.io/),
e configurar as variáveis de ambiente QDRANT_URL e QDRANT_API_KEY com as
credenciais do cluster no arquivo .env.
"""
# definição de constantes
load_dotenv()

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "financial"
FILE_PATH = "./AAPL_10-K_1A_temp.md"

# Criação do cliente do Qdrant
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

# Deleta a coleção (se necessário), e cria a coleção no Qdrant
if qdrant.get_collection(collection_name=COLLECTION_NAME):
    print(
        f"Collection '{COLLECTION_NAME}' já existe. \
          Deletando coleção existente."
    )
    qdrant.delete_collection(COLLECTION_NAME)

# Criação da coleção no Qdrant
print(f"Collection '{COLLECTION_NAME}' não existe. Criando nova coleção.")
qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=384,
        distance=models.Distance.COSINE,
    ),
)

# INGESTÃO E PREPROCESSAMENTO: Leitura do arquivo e criação dos chunks
with open(FILE_PATH, "r", encoding="utf-8") as f:
    content = f.read()

# Chuncks simples: cada parágrafo é um chunk
paragraphs = content.split("\n\n")
chunks = [p.strip() for p in paragraphs if len(p.strip()) > 50]

print(f"Total de chunks criados: {len(chunks)}")
print(f"Exemplo de chunk: '{chunks[0]}'")

# INDEXAÇÃO: criação dos embeddings e upload para o Qdrant
model = TextEmbedding(MODEL_NAME)

points = []
for chunk in chunks:
    embedding = list(model.passage_embed([chunk]))[0].tolist()
    point = models.PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding,
        payload={"text": chunk, "source": FILE_PATH},
    )
    points.append(point)

qdrant.upload_points(collection_name=COLLECTION_NAME, points=points)

# CONSULTA: criação do embedding da query e busca no Qdrant
query_text = "What are the main financial risks?"
query_embedding = list(model.query_embed([query_text]))[0].tolist()

# Consulta limitando a 3 resultados mais relevantes
results = qdrant.query_points(
    collection_name=COLLECTION_NAME,
    query=query_embedding,
    limit=3,
)

# Imprime os resultados mais relevantes da consulta
for r in results.points:
    print(f"Score: {r.score}")
    print(f"Texto: {r.payload['text'][:100]}...")
    print("-" * 80)
