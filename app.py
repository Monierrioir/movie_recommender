import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# GitHub Pages/Frontend alan adın geldikçe buraya eklersin (virgülle)
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173"
).split(",")

app = FastAPI(title="Movie Recommender API", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS if o],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"ok": True, "service": "movie-recommender-api"}

@app.get("/ping")
def ping():
    return {"pong": True}

@app.get("/healthz")
def health():
    return {"status": "healthy"}

