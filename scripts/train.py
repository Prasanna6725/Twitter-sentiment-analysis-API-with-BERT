"""
Model training script for sentiment analysis using BERT.
Fine-tunes a pre-trained BERT model on sentiment classification task.
"""
import json
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def load_data(train_path, test_path):
    """Load train and test data from CSV files."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def tokenize_function(examples, tokenizer, max_length=128):
    """Tokenize text data."""
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=max_length
    )

def main():
    """Train and evaluate a Transformer model for sentiment classification.

    The base model can be chosen via the `MODEL_NAME` environment variable
    (e.g. `bert-base-uncased` or `distilbert-base-uncased`).  `distilbert-base-uncased`
    is the default because it's smaller and trains faster, but you can switch to
    the original BERT by setting `MODEL_NAME=bert-base-uncased` before running
    the script.
    """
    import os

    print("Loading data...")
    train_df, test_df = load_data('data/processed/train.csv', 'data/processed/test.csv')
    
    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['text', 'label']])
    
    print("Loading tokenizer and model...")
    model_name = os.getenv('MODEL_NAME', 'distilbert-base-uncased')
    print(f"Using base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )
    
    print("Tokenizing data...")
    train_tokenized = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )
    test_tokenized = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )
    
    # Remove text column as we now have token_ids
    train_tokenized = train_tokenized.remove_columns(['text'])
    test_tokenized = test_tokenized.remove_columns(['text'])
    
    # Rename label column to match expected format
    train_tokenized = train_tokenized.rename_columns({'label': 'labels'})
    test_tokenized = test_tokenized.rename_columns({'label': 'labels'})
    
    # Set format for PyTorch
    train_tokenized.set_format('torch')
    test_tokenized.set_format('torch')
    
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir='./training_output',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
    )
    
    def compute_metrics(eval_pred):
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, zero_division=0),
            'recall': recall_score(labels, predictions, zero_division=0),
            'f1': f1_score(labels, predictions, zero_division=0),
        }
    
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        compute_metrics=compute_metrics,
    )
    
    print("Training model...")
    trainer.train()
    
    print("Evaluating model...")
    eval_results = trainer.evaluate()
    
    print("Saving model...")
    output_dir = Path('model_output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    # Save metrics
    results_dir = Path('results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = {
        'accuracy': float(eval_results.get('eval_accuracy', 0)),
        'precision': float(eval_results.get('eval_precision', 0)),
        'recall': float(eval_results.get('eval_recall', 0)),
        'f1_score': float(eval_results.get('eval_f1', 0)),
    }
    
    metrics_path = results_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save run summary
    run_summary = {
        'hyperparameters': {
            'model_name': model_name,
            'learning_rate': 5e-5,
            'batch_size': 16,
            'num_epochs': 3,
        },
        'final_metrics': {
            'accuracy': float(eval_results.get('eval_accuracy', 0)),
            'f1_score': float(eval_results.get('eval_f1', 0)),
        }
    }
    
    run_summary_path = results_dir / 'run_summary.json'
    with open(run_summary_path, 'w') as f:
        json.dump(run_summary, f, indent=2)
    
    print(f"Model saved to {output_dir}")
    print(f"Metrics saved to {metrics_path}")
    print(f"Run summary saved to {run_summary_path}")
    print("Training complete!")

if __name__ == '__main__':
    main()
