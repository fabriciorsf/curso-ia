# Aplicação Web de exemplo usando o FastAPI

1. Abra um terminal, navege até este diretório do projeto (`curso-ia\projeto\app`), e inicie a aplicação com o comando abaixo:

```bash
uvicorn main:app --reload
```

2. Abra outro terminal, e execute uma requisição à aplicação com um dos comandos abaixo:

- Linux:

```bash
curl -X 'POST' 'http://localhost:8000/events/' `
-H 'accept: application/json' `
-H 'Content-Type: application/json' `
-d '{
    "event_id": "123",
    "event_type": "live",
    "event_data": { "message": "xpto" }
}'
```

- Windows (PowerShell):

```bash
curl.exe -X 'POST' 'http://localhost:8000/events/' `
-H 'accept: application/json' `
-H 'Content-Type: application/json' `
-d '{
    \"event_id\": \"123\",
    \"event_type\": \"live\",
    \"event_data\": { \"message\": \"xpto\" }
}'
```
