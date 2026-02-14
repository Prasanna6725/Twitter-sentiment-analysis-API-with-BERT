#!/usr/bin/env python3
import re

POSITIVE_WORDS = {'good', 'great', 'excellent', 'amazing', 'love', 'loved', 'fantastic', 'best', 'wonderful', 'enjoyed', 'recommend', 'awesome', 'happy', 'perfect'}
NEGATIVE_WORDS = {'bad', 'terrible', 'awful', 'worst', 'hate', 'hated', 'boring', 'disappointing', 'poor', 'horrible', 'waste', 'unwatchable', 'sad', 'sadness', 'unhappy', 'disappointed', 'depressed', 'angry', 'annoyed', 'frustrated', 'upset'}

def rule_based_predict(text):
    txt = text.lower()
    pos = sum(1 for w in POSITIVE_WORDS if w in txt)
    neg = sum(1 for w in NEGATIVE_WORDS if w in txt)
    if pos == 0 and neg == 0:
        return 'neutral', 0.5
    if pos >= neg:
        confidence = pos / (pos + neg)
        return 'positive', float(confidence)
    else:
        confidence = neg / (pos + neg)
        return 'negative', float(confidence)

test_words = ['sad', 'happy', 'bad', 'good', 'i am sad today', 'this is great', 'sad and angry']
print(f"{'Input Text':<30} {'Sentiment':<12} {'Confidence':<12}")
print("=" * 54)
for word in test_words:
    sent, conf = rule_based_predict(word)
    print(f'{word:<30} {sent:<12} {conf:.2f}')
