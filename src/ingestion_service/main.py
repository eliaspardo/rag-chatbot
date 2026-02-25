from fastapi import FastAPI

from src.ingestion_service.lifespan import lifespan

app = FastAPI(lifespan=lifespan)


@app.get("/health")
def read_root():
    return {"status": "ok"}
