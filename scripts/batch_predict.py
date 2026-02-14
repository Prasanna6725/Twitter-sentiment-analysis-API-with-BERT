"""
Batch prediction script for sentiment analysis.
Reads a CSV file with text and generates predictions.
"""
import argparse
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model_and_tokenizer(model_path):
    """Load fine-tuned model and tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        return model, tokenizer
    except Exception:
        return None, None


# Rule-based fallback classifier
POSITIVE_WORDS = {"good", "great", "excellent", "amazing", "love", "loved", "fantastic", "best", "wonderful", "enjoyed", "recommend", "awesome"}
NEGATIVE_WORDS = {"bad", "terrible", "awful", "worst", "hate", "hated", "boring", "disappointing", "poor", "horrible", "waste", "unwatchable"}

def rule_based_predict_text(text):
    txt = text.lower()
    pos = sum(1 for w in POSITIVE_WORDS if w in txt)
    neg = sum(1 for w in NEGATIVE_WORDS if w in txt)
    if pos == 0 and neg == 0:
        return {'sentiment': 'neutral', 'confidence': 0.5}
    if pos >= neg:
        confidence = pos / (pos + neg)
        return {'sentiment': 'positive', 'confidence': float(confidence)}
    else:
        confidence = neg / (pos + neg)
        return {'sentiment': 'negative', 'confidence': float(confidence)}

def predict_sentiment(texts, model, tokenizer):
    """Predict sentiment for a batch of texts."""
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    predictions = torch.argmax(logits, dim=1)
    
    sentiments = ['negative', 'positive']
    results = []
    
    for pred_idx, prob in zip(predictions, probabilities):
        sentiment = sentiments[pred_idx.item()]
        confidence = prob[pred_idx].item()
        results.append({
            'sentiment': sentiment,
            'confidence': confidence
        })
    
    return results

def main():
    """Run batch predictions on CSV file."""
    parser = argparse.ArgumentParser(description='Batch sentiment prediction')
    parser.add_argument('--input-file', required=True, help='Input CSV file path')
    parser.add_argument('--output-file', required=True, help='Output CSV file path')
    parser.add_argument('--model-path', default='model_output', help='Path to fine-tuned model')
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    
    print(f"Reading input file {args.input_file}...")
    df = pd.read_csv(args.input_file)
    
    if 'text' not in df.columns:
        raise ValueError("Input CSV must have a 'text' column")
    
    print(f"Processing {len(df)} texts...")
    if model is not None and tokenizer is not None:
        predictions = predict_sentiment(df['text'].tolist(), model, tokenizer)
        df['predicted_sentiment'] = [p['sentiment'] for p in predictions]
        df['confidence'] = [p['confidence'] for p in predictions]
    else:
        # Use rule-based fallback
        preds = [rule_based_predict_text(t) for t in df['text'].tolist()]
        df['predicted_sentiment'] = [p['sentiment'] for p in preds]
        df['confidence'] = [p['confidence'] for p in preds]
    
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == '__main__':
    main()
