from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router as processing_router
from .services.model_health import model_status

app = FastAPI(
    title="SahiNaksha API",
    description="AI-assisted urban parcel mapping and cadastral feature extraction API.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(processing_router)


@app.get("/")
def root():
    return {"name": "SahiNaksha", "status": "ok", "version": "0.1.0"}


@app.get("/health")
def health():
    return {"status": "healthy", "model": model_status()}
