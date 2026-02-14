#!/usr/bin/env python3
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

print('Loading model...')
model = AutoModelForSequenceClassification.from_pretrained('model_output_quantized')
tokenizer = AutoTokenizer.from_pretrained('model_output_quantized')

test_texts = ['sad', "don't like", 'happy', 'good', 'bad']

print(f"{'Text':<20} {'Model Pred':<12} {'Confidence':<12} {'Logits':<40}")
print("=" * 85)

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
    sentiments = ['negative', 'positive']
    sentiment = sentiments[prediction]
    
    logits_str = f"[{logits[0][0].item():.3f}, {logits[0][1].item():.3f}]"
    print(f'{text:<20} {sentiment:<12} {confidence:<12.3f} {logits_str:<40}')
