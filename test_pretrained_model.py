#!/usr/bin/env python3
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

print('Loading pre-trained model...')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

sentiments = ['negative', 'positive']

test_texts = [
    'sad',
    "don't like", 
    'happy',
    'good',
    'bad',
    'i am very sad',
    'this is great',
    'terrible movie',
    'I absolutely loved this movie!',
    'This was the worst experience ever'
]

print(f"{'Text':<35} {'Prediction':<12} {'Confidence':<12}")
print("=" * 60)

for text in test_texts:
    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    prediction = torch.argmax(logits, dim=1).item()
    confidence = probabilities[0][prediction].item()
    sentiment = sentiments[prediction]
    
    print(f'{text:<35} {sentiment:<12} {confidence:<12.3f}')
