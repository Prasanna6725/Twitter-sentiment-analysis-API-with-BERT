#!/usr/bin/env python3
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

print('Loading model...')
model = AutoModelForSequenceClassification.from_pretrained('model_output_quantized')
tokenizer = AutoTokenizer.from_pretrained('model_output_quantized')

# Corrected labels: 0=positive, 1=negative (inverted)
sentiments = ['positive', 'negative']

test_texts = ['sad', "don't like", 'happy', 'good', 'bad', 'i am very sad', 'this is great', 'terrible movie']

print(f"{'Text':<25} {'Prediction':<12} {'Confidence':<12}")
print("=" * 50)

for text in test_texts:
    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='pt')
    if 'token_type_ids' in inputs:
        inputs.pop('token_type_ids')
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    prediction = torch.argmax(logits, dim=1).item()
    confidence = probabilities[0][prediction].item()
    sentiment = sentiments[prediction]
    
    print(f'{text:<25} {sentiment:<12} {confidence:<12.3f}')
