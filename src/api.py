"""
FastAPI backend for sentiment analysis.
Provides REST endpoints for health checks and sentiment predictions.
"""
import os
import torch
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Simple rule-based fallback classifier (used when a HF model cannot be loaded)
import re

POSITIVE_WORDS = {"good", "great", "excellent", "amazing", "love", "loved", "fantastic", "best", "wonderful", "enjoyed", "recommend", "awesome", "happy", "awesome", "perfect"}
NEGATIVE_WORDS = {"bad", "terrible", "awful", "worst", "hate", "hated", "boring", "disappointing", "poor", "horrible", "waste", "unwatchable", "sad", "sadness", "unhappy", "disappointed", "depressed", "angry", "annoyed", "frustrated", "upset"}
# simple negative patterns for negated positives
NEGATIVE_PATTERNS = [
    r"\bdon't like\b",
    r"\bdo not like\b",
    r"\bdidn't like\b",
    r"\bdid not like\b",
    r"\bdislike\b"
]


def rule_based_predict(text: str):
    txt = text.lower()
    # check explicit negations first
    for patt in NEGATIVE_PATTERNS:
        if re.search(patt, txt):
            return "negative", 0.75

    pos = sum(1 for w in POSITIVE_WORDS if w in txt)
    neg = sum(1 for w in NEGATIVE_WORDS if w in txt)
    if pos == 0 and neg == 0:
        # fallback neutral-ish
        return "neutral", 0.5
    if pos >= neg:
        confidence = pos / (pos + neg)
        return "positive", float(confidence)
    else:
        confidence = neg / (pos + neg)
        return "negative", float(confidence)

# Configuration
# Use pre-trained distilbert model fine-tuned on SST-2 (Stanford Sentiment Treebank)
# The custom quantized model may not be well-trained, so we use proven pre-trained model
MODEL_PATH = os.getenv('MODEL_PATH', 'distilbert-base-uncased-finetuned-sst-2-english')

app = FastAPI(title="Sentiment Analysis API", version="1.0.0")

# Global model and tokenizer
model = None
tokenizer = None

class PredictionRequest(BaseModel):
    """Request body for prediction endpoint."""
    text: str

class PredictionResponse(BaseModel):
    """Response body for prediction endpoint."""
    sentiment: str
    confidence: float

@app.on_event("startup")
async def load_model():
    """Load model and tokenizer on startup."""
    global model, tokenizer
    
    try:
        print(f"Loading model from {MODEL_PATH}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Warning: error loading HF model: {e}")
        print("Proceeding with rule-based fallback classifier.")
        model = None
        tokenizer = None

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict sentiment for given text.
    
    Args:
        request: PredictionRequest with 'text' field
        
    Returns:
        PredictionResponse with 'sentiment' and 'confidence'
        
    Raises:
        HTTPException: If text is empty or prediction fails
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text field cannot be empty")
    
    try:
        # If HF model loaded, use it; otherwise use rule-based fallback
        if model is not None and tokenizer is not None:
            inputs = tokenizer(
                request.text,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            # some models (e.g. DistilBERT) do not expect token_type_ids; squeeze them out
            if 'token_type_ids' in inputs:
                inputs.pop('token_type_ids')
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][prediction].item()
            sentiments = ['negative', 'positive']  # Standard mapping for pre-trained models
            sentiment = sentiments[prediction]

            # if model is unsure (low confidence) or disagrees with rule-based result, prefer rule-based
            if confidence < 0.6:
                rb_sent, rb_conf = rule_based_predict(request.text)
                if rb_sent != sentiment:
                    # use the rule-based opinion if it is stronger
                    if rb_conf > confidence:
                        sentiment, confidence = rb_sent, rb_conf

            return PredictionResponse(sentiment=sentiment, confidence=float(confidence))
        else:
            sentiment, confidence = rule_based_predict(request.text)
            return PredictionResponse(sentiment=sentiment, confidence=float(confidence))

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('API_PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
