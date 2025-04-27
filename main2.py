import os
import io
import logging
from typing import List

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import uvicorn

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Lazy-loaded model
model = None

def load_model():
    global model
    if model is None:
        logger.info("Loading SentenceTransformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Model loaded successfully.")
    return model

# Response Models
class MatchResult(BaseModel):
    match: str
    category_id: int = Field(..., alias="category_id")
    score: float
    confidence: float

    class Config:
        allow_population_by_field_name = True
        orm_mode = True


class CustomerMatch(BaseModel):
    customer_category: str = Field(..., alias="customer_category")
    matches: List[MatchResult]

    class Config:
        allow_population_by_field_name = True
        orm_mode = True


class FullMatchResponse(BaseModel):
    matches: List[CustomerMatch]

    class Config:
        allow_population_by_field_name = True
        orm_mode = True


# Main endpoint
@app.post("/upload_and_match", response_model=FullMatchResponse)
async def match_categories(file: UploadFile = File(...), customerCategories: List[str] = None, topN: int = 5):
    try:
        logger.debug("Received match request.")

        if not customerCategories or not file:
            raise HTTPException(status_code=400, detail="Missing required parameters")

        model_instance = load_model()

        file_content = await file.read()
        df = pd.read_csv(io.BytesIO(file_content), encoding="ISO-8859-1")

        if 'id' not in df.columns or 'path' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'id' and 'path' columns.")

        category_paths = df['path'].astype(str).tolist()
        category_embeddings = model_instance.encode(category_paths, convert_to_tensor=True)

        results = []
        for raw_category in customerCategories:
            clean_category = " > ".join(part.strip() for part in raw_category.split(">"))
            input_embedding = model_instance.encode(clean_category, convert_to_tensor=True)
            cos_scores = util.pytorch_cos_sim(input_embedding, category_embeddings)[0]
            top_results = cos_scores.topk(min(topN, len(category_paths)))

            matches = [
                MatchResult(
                    match=category_paths[idx],
                    category_id=int(df['id'].iloc[idx]),
                    score=round(float(score), 4),
                    confidence=round(float(score) * 100, 2),
                )
                for score, idx in zip(top_results.values, top_results.indices)
            ]

            results.append(CustomerMatch(
                customer_category=clean_category,
                matches=matches
            ))

        return FullMatchResponse(matches=results)

    except Exception as e:
        logger.error(f"Error during matching: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Health endpoint
@app.get("/health")
async def health_check():
    return {"status": "OK", "service": "SBERT Matcher"}


# Startup configuration
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Starting server on port {port}...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=False,
        workers=1,
        timeout_keep_alive=60
    )
