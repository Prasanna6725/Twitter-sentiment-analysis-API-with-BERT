#!/usr/bin/env python3
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

print('Loading model...')
model = AutoModelForSequenceClassification.from_pretrained('model_output_quantized')
tokenizer = AutoTokenizer.from_pretrained('model_output_quantized')

# Test with actual examples from training data
# From train.csv: label 0 = positive reviews, label 1 = negative reviews
test_cases = [
    ('sad', 'negative'),
    ("don't like", 'negative'),
    ('happy', 'positive'),
    ('This is by far one of the worst movies i have ever seen', 'negative'),  # From label 0 row - WAIT this should be label 1
    ('Having a close experience with one such patient is probably the best reason', 'positive'),  # label 1 - wait
]

# Check actual labels in data
print("\nChecking what labels in CSV mean:\n")
with open('data/processed/train.csv', 'r', encoding='utf-8') as f:
    lines = f.readlines()[:3]  # First 3 lines
    for line in lines:
        if line.startswith('text'):
            print("HEADER:", line[:50])
        parts = line.split(',', 1)
        if len(parts) == 2:
            text = parts[0][:60]
            label = parts[1].strip()
            print(f"Label {label}: {text}...")

print("\n" + "="*70)
print("Testing sentiment predictions:\n")

# Try both label orders
for label_order in [['positive', 'negative'], ['negative', 'positive']]:
    print(f"\nUsing label order: {label_order}")
    print(f"{'Text':<35} {'Prediction':<12} {'Confidence':<12}")
    print("-" * 60)
    
    for text, expected in test_cases[:3]:
        inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='pt')
        if 'token_type_ids' in inputs:
            inputs.pop('token_type_ids')
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0][prediction].item()
        sentiment = label_order[prediction]
        match = "✓" if sentiment == expected else "✗"
        
        print(f'{text:<35} {sentiment:<12} {confidence:<12.3f} {match}')
