import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import uvicorn
import logging
from typing import List
import numpy as np
import io

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"]  # Expose all headers
)

# Load ML model
try:
    logger.info("Loading SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
    raise

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

# Explicit OPTIONS handler for CORS preflight
@app.options("/upload_and_match")
async def options_upload_match():
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )

# Main endpoint
@app.post("/upload_and_match", response_model=FullMatchResponse)
async def match_categories(file: UploadFile = File(...), customerCategories: List[str] = None, topN: int = 5):
    try:
        logger.debug("Received request for category matching...")

        if not customerCategories or not file:
            raise HTTPException(status_code=400, detail="Missing file or customer categories")

        file_content = await file.read()
        df = pd.read_csv(io.BytesIO(file_content), encoding="ISO-8859-1")

        if df.empty:
            raise ValueError("CSV file is empty")

        category_paths = df['path'].astype(str).tolist()
        category_ids = df['id'].astype(int).tolist()

        logger.info("Generating embeddings for categories...")
        category_embeddings = model.encode(category_paths, convert_to_tensor=True)

        results = []
        for raw_category in customerCategories:
            clean_category = " > ".join(part.strip() for part in raw_category.split(">") if part.strip())
            input_embedding = model.encode(clean_category, convert_to_tensor=True)
            cos_scores = util.pytorch_cos_sim(input_embedding, category_embeddings)[0]
            top_results = cos_scores.topk(min(topN, len(category_paths)))

            matched = []
            for score, idx in zip(top_results.values, top_results.indices):
                matched.append(MatchResult(
                    match=category_paths[idx],
                    category_id=int(category_ids[idx]),
                    score=round(float(score), 4),
                    confidence=round(float(score) * 100, 2)
                ))

            results.append(CustomerMatch(
                customer_category=clean_category,
                matches=matched
            ))

        return FullMatchResponse(matches=results)

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "SBERT Category Matcher"}

# Startup configuration
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render default port
    uvicorn.run(
        "main2:app",
        host="0.0.0.0",  # Must bind to 0.0.0.0 for Render
        port=port,
        reload=False,  # Disable auto-reload in production
        workers=1,  # Single worker for Render free tier
        timeout_keep_alive=60  # Keep connections alive longer
    )