import os
from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# .env dosyasını oku (varsa)
load_dotenv()

TMDB_KEY = os.getenv("TMDB_API_KEY", "")

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173"
).split(",")

app = FastAPI(title="Movie Recommender API", version="0.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS if o],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Sağlık ----------
@app.get("/")
def root():
    return {"ok": True, "service": "movie-recommender-api"}

@app.get("/ping")
def ping():
    return {"pong": True}

@app.get("/healthz")
def health():
    return {"status": "healthy"}

# ---------- TMDB yardımcıları ----------
def tmdb_search_movie(title: str):
    if not TMDB_KEY:
        return None
    r = requests.get(
        "https://api.themoviedb.org/3/search/movie",
        params={"api_key": TMDB_KEY, "query": title, "language": "tr-TR"}
    )
    r.raise_for_status()
    arr = r.json().get("results", [])
    return arr[0] if arr else None

def tmdb_movie_details(mid: int):
    r = requests.get(
        f"https://api.themoviedb.org/3/movie/{mid}",
        params={"api_key": TMDB_KEY, "append_to_response": "credits", "language": "tr-TR"}
    )
    r.raise_for_status()
    return r.json()

def enrich_movie_by_title(title: str) -> Optional[dict]:
    m = tmdb_search_movie(title)
    if not m:
        return None
    d = tmdb_movie_details(m["id"])
    genres = [g["name"] for g in d.get("genres", [])]
    crew = d.get("credits", {}).get("crew", [])
    cast = d.get("credits", {}).get("cast", [])
    directors = [c["name"] for c in crew if c.get("job") == "Director"]
    top_cast = [c["name"] for c in cast[:5]]
    return {
        "id": d["id"],
        "title": d.get("title") or title,
        "year": (d.get("release_date") or "")[:4],
        "genres": genres,
        "directors": directors,
        "cast": top_cast,
        "overview": d.get("overview") or "",
        "popularity": float(d.get("popularity", 0.0)),
        "vote": float(d.get("vote_average", 0.0)),
    }

def movie_profile_text(m: dict) -> str:
    return " | ".join([
        m.get("overview",""),
        "Türler: " + ", ".join(m.get("genres", [])),
        "Oyuncular: " + ", ".join(m.get("cast", [])),
        "Yönetmen: " + ", ".join(m.get("directors", [])),
        f"Yıl: {m.get('year','')}"
    ])

# ---------- Basit DB + embedding modeli ----------
class DB:
    def __init__(self):
        self.df = pd.DataFrame()
        self.emb = None
        # küçük ve hızlı çok dilli model
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def ingest(self, movies: List[dict]):
        rows = []
        for m in movies:
            if not m: continue
            r = {
                "id": m.get("id"),
                "title": m.get("title"),
                "year": m.get("year"),
                "genres": m.get("genres", []),
                "directors": m.get("directors", []),
                "cast": m.get("cast", []),
                "overview": m.get("overview",""),
                "popularity": m.get("popularity",0.0),
                "vote": m.get("vote",0.0),
            }
            r["profile_text"] = movie_profile_text(r)
            rows.append(r)
        if not rows:
            return
        df_new = pd.DataFrame(rows).drop_duplicates(subset=["id","title"])
        # mevcutla birleştir
        self.df = pd.concat([self.df, df_new], ignore_index=True).drop_duplicates(subset=["id","title"])
        # embedding’leri yeniden hesapla (küçük veri için pratik)
        self.emb = self.model.encode(self.df["profile_text"].tolist(), normalize_embeddings=True)

    def similar(self, liked: List[str], top_k=10):
        if self.df.empty or self.emb is None:
            return []
        liked_rows = self.df[self.df["title"].str.lower().isin([t.lower() for t in liked])]
        if liked_rows.empty:
            return []
        user_vec = self.emb[liked_rows.index].mean(axis=0, keepdims=True)
        sims = cosine_similarity(user_vec, self.emb)[0]
        score = sims \
            + 0.02*(self.df["vote"].fillna(0).to_numpy()/10.0) \
            + 0.01*(self.df["popularity"].fillna(0).to_numpy()/(self.df["popularity"].max() or 1.0))
        rec = self.df.copy()
        rec["score"] = score
        rec = rec[~rec["title"].str.lower().isin([t.lower() for t in liked])]
        rec = rec.sort_values("score", ascending=False).head(top_k)
        return rec[["title","year","genres","directors","score"]].to_dict(orient="records")

db = DB()

# ---------- API şemaları ----------
class SeedIn(BaseModel):
    titles: List[str]

class RecommendIn(BaseModel):
    liked_titles: List[str]
    top_k: int = 10

# ---------- Uçlar ----------
@app.post("/ingest")
def ingest(payload: SeedIn):
    """Film başlıklarını TMDB'den zenginleştirip hafızaya al."""
    movies = []
    for t in payload.titles:
        try:
            movies.append(enrich_movie_by_title(t))
        except Exception:
            movies.append(None)
    db.ingest(movies)
    ok = len([m for m in movies if m])
    return {"ingested": ok}

@app.post("/recommend")
def recommend(payload: RecommendIn):
    """Beğenilen başlıklara göre benzer film önerileri döndür."""
    out = db.similar(payload.liked_titles, top_k=payload.top_k)
    return {"results": out}
