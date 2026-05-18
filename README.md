# Material da especialização de Engenharia de IA | Dev + Eficiente

## Configuração do Ambiente Python

### Preparando o Interpretador Python

```bash
py list
py install 3.13 --update    ## o python 3.14 ainda está incompatível com o SparseTextEmbedding do fastembed (dep: qdrant-client[fastembed])
```

### Instalação do Gerenciador de Ambiente `uv`

- Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv self update
```

- Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
$env:Path = "$HOME\.local\bin;$env:Path"
uv self update
```

### Ativação do ambiente

```bash
.venv\Scripts\activate
```

### Instalando as versões atuais de todas as bibliotecas instaladas no ambiente

```bash
uv sync
uv python pin 3.13
uv lock --upgrade
uv sync --upgrade
```

### Listar todas as bibliotecas instaladas no ambiente

```bash
uv pip list
```

### Instalando dependências principais (caso ainda não estejam instalados)

```bash
## Numpy
uv add numpy

## Grok
uv add groq

## OpenIA
uv add openai

## Sentences Transformers
uv add sentence-transformers

## Qdrant
uv add qdrant-client

## Docling
uv add docling

## LangExtract
uv add langextract

## FastEmbed (ONNX)
uv add fastembed
### or with GPU support
uv add fastembed-gpu

## python-dotenv
uv add python-dotenv

## HDBSCAN
uv add hdbscan

## EdgarTools
uv add edgartools

## FastAPI
uv add fastapi

## uvicorn
uv add Uvicorn
```
