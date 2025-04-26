import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import uvicorn
import logging
from typing import List
import numpy as np
import io

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

origins = [
    "http://localhost:4200",  # Your local Angular app
    "https://sbert-microservice.onrender.com",  # Your deployed Render app
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow requests from both origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

try:
    logger.info("Loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    logger.info("Model loaded successfully.")

except Exception as e:
    logger.error(f"Initialization failed: {str(e)}")
    raise


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

        logger.debug(f"First 3 categories: {category_paths[:3]}")
        logger.debug(f"First 3 IDs: {category_ids[:3]}")

        logger.info("Generating embeddings for categories...")
        category_embeddings = model.encode(category_paths, convert_to_tensor=True)

        if len(category_embeddings) != len(category_paths):
            raise ValueError("Embeddings generation failed")

        logger.info(f"Successfully loaded {len(category_paths)} categories from CSV.")

        results = []

        for raw_category in customerCategories:
            clean_category = " > ".join(part.strip() for part in raw_category.split(">") if part.strip())
            logger.debug(f"Processing category: '{raw_category}' -> '{clean_category}'")

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

            logger.debug(f"Found {len(matched)} matches for '{clean_category}'")
            results.append(CustomerMatch(
                customer_category=clean_category,
                matches=matched
            ))

        logger.info(f"Returning {len(results)} result groups")
        return FullMatchResponse(matches=results)

    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main2:app", host="0.0.0.0", port=port, log_level="debug")