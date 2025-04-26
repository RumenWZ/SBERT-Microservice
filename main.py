import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import uvicorn
import logging
from typing import List
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Changed to DEBUG for more details
logger = logging.getLogger(__name__)

app = FastAPI()

# Load model and data with validation
try:
    logger.info("Loading model and data...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Read CSV with additional validation
    df = pd.read_csv("amazon_categories.csv", encoding="ISO-8859-1")
    if df.empty:
        raise ValueError("CSV file is empty")

    # Convert and validate columns
    category_paths = df['path'].astype(str).tolist()
    category_ids = df['id'].astype(int).tolist()

    logger.debug(f"First 3 categories: {category_paths[:3]}")
    logger.debug(f"First 3 IDs: {category_ids[:3]}")

    # Generate embeddings with validation
    logger.info("Generating embeddings...")
    category_embeddings = model.encode(category_paths, convert_to_tensor=True)
    if len(category_embeddings) != len(category_paths):
        raise ValueError("Embeddings generation failed")

    logger.info(f"Successfully loaded {len(category_paths)} categories")

except Exception as e:
    logger.error(f"Initialization failed: {str(e)}")
    raise


# Request/Response Models
class MatchedCategory(BaseModel):
    match: str
    categoryId: int
    score: float
    confidence: float


class CategoryMatches(BaseModel):
    customerCategory: str
    matches: List[MatchedCategory]


class MatchResponse(BaseModel):
    results: List[CategoryMatches]


class MatchRequest(BaseModel):
    customerCategories: List[str]
    topN: int = 5


@app.post("/match", response_model=MatchResponse)
async def match_categories(request: MatchRequest):
    try:
        logger.debug(f"Received request: {request.dict()}")

        results = []

        for raw_category in request.customerCategories:
            # Clean and normalize the category string
            clean_category = " > ".join(
                part.strip()
                for part in raw_category.split(">")
                if part.strip()
            )

            logger.debug(f"Processing category: '{raw_category}' -> '{clean_category}'")

            # Generate embedding
            input_embedding = model.encode(clean_category, convert_to_tensor=True)

            # Calculate similarities
            cos_scores = util.pytorch_cos_sim(input_embedding, category_embeddings)[0]

            # Get top matches
            top_results = cos_scores.topk(min(request.topN, len(category_paths)))

            # Prepare matches
            matched = []
            for score, idx in zip(top_results.values, top_results.indices):
                matched.append(MatchedCategory(
                    match=category_paths[idx],
                    categoryId=int(category_ids[idx]),
                    score=round(float(score), 4),
                    confidence=round(float(score) * 100, 2)
                ))

            logger.debug(f"Found {len(matched)} matches for '{clean_category}'")
            results.append(CategoryMatches(
                customerCategory=clean_category,
                matches=matched
            ))

        logger.info(f"Returning {len(results)} result groups")
        return MatchResponse(results=results)

    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="debug")