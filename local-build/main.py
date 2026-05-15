from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

MODEL_VERSION = "v2"

app = FastAPI(title="DistilBERT Sentiment API")

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    score: float
    model_version: str


@app.get("/health")
def health():
    return {"status": "ok", "version": MODEL_VERSION}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")
    result = classifier(request.text)[0]
    return PredictResponse(
        label=result["label"],
        score=round(result["score"], 4),
        model_version=MODEL_VERSION
    )
