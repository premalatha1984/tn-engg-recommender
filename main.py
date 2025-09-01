from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
from typing import List
from models import RecommendationRequest, OptionsResponse, Weights
from algorithm import load_data, recommend

DATA_DIR = "data"

app = FastAPI(title="TN Engineering Recommender", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

colleges_df, programs_df, cutoffs_df = load_data(DATA_DIR)


@app.get("/")
def read_root():
    return {"Hello": "World"}
@app.get("/api/weights/default")
def get_default_weights():
    return Weights()

@app.get("/api/options", response_model=OptionsResponse)
def get_options():
    categories = ["OC", "BC", "MBC", "SC", "ST"]
    branches = sorted(programs_df["branch"].unique().tolist())
    districts = sorted(colleges_df["district"].unique().tolist())
    return OptionsResponse(
        categories=categories,
        branches=branches,
        districts=districts,
        default_weights=Weights()
    )

@app.post("/api/recommendations")
def recommendations(req: RecommendationRequest):
    weights = req.weights or Weights()
    results = recommend(req.profile, weights, colleges_df, programs_df, cutoffs_df)
    return JSONResponse(results)
