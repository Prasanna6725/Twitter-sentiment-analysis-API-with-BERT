"""
Data preprocessing script for sentiment analysis.
Loads IMDB dataset, cleans it, and splits into training and testing sets.
"""
import os
import re
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def clean_text(text):
    """Clean and normalize text."""
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    """Load dataset, preprocess, and save to CSV files."""
    print("Loading IMDB dataset...")
    dataset = load_dataset('imdb')
    
    # Combine train and test, then resplit for consistency
    train_data = dataset['train']
    test_data = dataset['test']
    
    # Create dataframe
    data_list = []
    
    for sample in train_data:
        data_list.append({
            'text': sample['text'],
            'label': sample['label']
        })
    
    for sample in test_data:
        data_list.append({
            'text': sample['text'],
            'label': sample['label']
        })
    
    df = pd.DataFrame(data_list)
    
    print("Cleaning text...")
    df['text'] = df['text'].apply(clean_text)
    
    # Remove empty texts
    df = df[df['text'].str.len() > 0]
    
    print("Splitting data into train and test sets...")
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )
    
    # Create output directory
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    train_path = output_dir / 'train.csv'
    test_path = output_dir / 'test.csv'
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Training data: {len(train_df)} samples saved to {train_path}")
    print(f"Test data: {len(test_df)} samples saved to {test_path}")
    print("Preprocessing complete!")

if __name__ == '__main__':
    main()
