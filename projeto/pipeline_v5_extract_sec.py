import os
import uuid

from dotenv import load_dotenv
from fastembed import (
    LateInteractionTextEmbedding,
    SparseTextEmbedding,
    TextEmbedding,
)
from qdrant_client import QdrantClient, models
from utils.edgar_client import EdgarClient
from utils.semantic_chunker import SemanticChunker

""" PIPELINE COMPLETO (com busca híbrida e re-ranking):
1. Configurações e inicialização do banco de vetores;
2. Ingestão e pré-processamento dos dados;
    - A ingestão é feita utilizando o EdgarClient para extrair os dados do SEC
    - O pré-processamento é feito utilizando o SemanticChunker
    - O SemanticChunker agrupa parágrafos em chunks semânticos
3. Indexação dos dados no banco de vetores;
    - A Indexação é feita utilizando tanto embeddings densos e esparsos
    - E armazena ambos os tipos de vetores no banco vetorial
4. Consulta e recuperação de informações relevantes.
    - A Consulta é feita utilizando uma query híbrida
    - Combinando os resultados de busca densa e esparsa
    - A combinação é realizada com o método RRF (Reciprocal Rank Fusion)
    - O re-ranking é realizado para melhorar a relevância dos resultados finais
    - O re-ranking é feito com o LLM ColBERT, gerando embeddings de tokens
    - O ColBERT busca a máximas similaridades entre os embeddings
      dos tokens das consultas diante os embeddings dos tokens dos chuncks
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

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL = "Qdrant/bm25"
COLBERT_MODEL = "colbert-ir/colbertv2.0"
COLLECTION_NAME = "financial"
EMAIL = os.getenv("EMAIL")
MAX_TOKENS = 300

# Criação do cliente do Qdrant
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

# Deleta a coleção (se necessário), e cria a coleção no Qdrant
if qdrant.get_collection(collection_name=COLLECTION_NAME):
    print(f"Collection '{COLLECTION_NAME}' já existe. Deletando coleção existente.")
    qdrant.delete_collection(COLLECTION_NAME)

# Criação da coleção no Qdrant
print(f"Collection '{COLLECTION_NAME}' não existe. Criando nova coleção.")
qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "dense": models.VectorParams(
            size=384,
            distance=models.Distance.COSINE,
        ),
        "colbert": models.VectorParams(
            size=128,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
        ),
    },
    sparse_vectors_config={
        "sparse": models.SparseVectorParams(),
    },
)

# INGESTÃO E PREPROCESSAMENTO: Extração dos dados do SEC usando o EdgarClient
edgar = EdgarClient(email=EMAIL)

data_10k = edgar.fetch_filing_data("AAPL", "10-K")
text_10k = edgar.get_combined_text(data_10k)

data_10q = edgar.fetch_filing_data("AAPL", "10-Q")
text_10q = edgar.get_combined_text(data_10q)

# Chuncks semânticos utilizando o SemanticChunker, que agrupa parágrafos
chunker = SemanticChunker(max_tokens=MAX_TOKENS)

all_chunks = []
docs = [(data_10k, text_10k), (data_10q, text_10q)]
for idx, (data, text) in enumerate(docs):
    print(f"Processando documento {idx + 1} de {len(docs)} ...")
    chunks = chunker.create_chunks(text)
    for chunk in chunks:
        all_chunks.append({"text": chunk, "metadata": data["metadata"]})

print(f"Total de chunks criados: {len(all_chunks)}")
print(f"Exemplo de chunk: '{all_chunks[0]['text'][:100]}...'")

# INDEXAÇÃO: criação dos embeddings e upload para o Qdrant
print("Criando embeddings densos...")
dense_model = TextEmbedding(DENSE_MODEL)
print("Criando embeddings esparsos...")
sparse_model = SparseTextEmbedding(SPARSE_MODEL)
print("Criando embeddings ColBERT...")
colbert_model = LateInteractionTextEmbedding(COLBERT_MODEL)

print("Criando pontos para upload...")
points = []
for idx, chunk_data in enumerate(all_chunks):
    chunk = chunk_data["text"]
    metadata = chunk_data["metadata"]

    dense_embedding = list(dense_model.passage_embed([chunk]))[0].tolist()
    sparse_embedding = list(sparse_model.passage_embed([chunk]))[0].as_object()
    colbert_embedding = list(colbert_model.passage_embed([chunk]))[0].tolist()

    point = models.PointStruct(
        id=str(uuid.uuid4()),
        vector={
            "dense": dense_embedding,
            "sparse": sparse_embedding,
            "colbert": colbert_embedding,
        },
        payload={
            "idx": idx,
            "text": chunk,
            "metadata": metadata,
        },
    )
    points.append(point)

print("Uploading points to Qdrant...")
qdrant.upload_points(
    collection_name=COLLECTION_NAME,
    points=points,
    batch_size=10,  # para evitar timeouts em uploads grandes
)


# CONSULTA: criação do embedding da query e busca no Qdrant
query_text = "What are the main financial risks?"
query_dense = list(dense_model.query_embed([query_text]))[0].tolist()
query_sparse = list(sparse_model.query_embed([query_text]))[0].as_object()
query_colbert = list(colbert_model.query_embed([query_text]))[0].tolist()

# Visualizando os 3 vetores de consulta
print(f"Vetor de consulta denso: {query_dense}")
print(f"Vetor de consulta esparsa: {query_sparse}")
print(f"Vetor de consulta ColBERT: {query_colbert}")

# Consulta híbrida (densa + esparsa) limitando a 3 resultados mais relevantes
results = qdrant.query_points(
    collection_name=COLLECTION_NAME,
    prefetch=[
        {
            "prefetch": [
                {  # Busca densa
                    "query": query_dense,
                    "using": "dense",
                    "limit": 10,
                },
                {  # Busca esparsa
                    "query": query_sparse,
                    "using": "sparse",
                    "limit": 10,
                },
            ],
            # Utiliza o método RRF para combinar os resultados da busca híbrida
            "query": models.FusionQuery(fusion=models.Fusion.RRF),
            "limit": 20,  # busca mais resultados para o re-ranking com ColBERT
        }
    ],
    # Utiliza o ColBERT para re-ranquear os resultados da busca híbrida
    query=query_colbert,
    using="colbert",
    limit=3,
)

# Imprime os resultados mais relevantes da consulta com os scores absolutos
for r in results.points:
    print(f"Score: {r.score}")
    print(f"Texto: {r.payload['text'][:100]}...")
    print("-" * 80)

# Imprime os resultados mais relevantes da consulta com os scores normalizados
max_score = max(r.score for r in results.points) if results.points else 1.0
for r in results.points:
    normalized_score = r.score / max_score if max_score else 0
    print(f"Score: {normalized_score}")
    print(f"Texto: {r.payload['text'][:100]}...")
    print("-" * 80)
