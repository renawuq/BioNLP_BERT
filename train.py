import json
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from evaluate import load
from evaluate import EvaluationModule
import numpy as np
import torch
import argparse
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(jsonl_file, max_samples=None):
    # Load all data first
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line")

    # Filter valid entries
    label2id = {'BioNLP': 1, 'Non_BioNLP': 0}
    filtered_data = []
    
    for item in data:
        if 'title' in item and 'abstract' in item:
            # Normalize labels
            if 'label' in item and item['label'] in [0, 1]:
                item['is_bionlp'] = 'BioNLP' if item['label'] == 1 else 'Non_BioNLP'
            elif 'is_bionlp' in item and item['is_bionlp'] in label2id:
                pass
            else:
                continue
                
            item['label'] = label2id[item['is_bionlp']]
            filtered_data.append(item)

    if not filtered_data:
        raise ValueError("No valid entries found in the dataset.")

    # Convert to DataFrame
    df = pd.DataFrame(filtered_data)
    
    # Randomly sample if max_samples is specified
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)  # Fixed random state for reproducibility
    
    # Combine text fields
    df['text'] = 'Title: ' + df['title'].fillna('') + '\nAbstract: ' + df['abstract'].fillna('')

    # Stratified train-test split
    if 'split' not in df.columns:
        train_df, dev_df = train_test_split(
            df, 
            test_size=0.2, 
            random_state=42, 
            stratify=df['label']
        )
        train_df['split'] = 'train'
        dev_df['split'] = 'dev'
        df = pd.concat([train_df, dev_df])

    return Dataset.from_pandas(df[['text', 'label', 'split']]), label2id

def tokenize_dataset(dataset, tokenizer_name):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Preprocess function
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)
    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    return tokenized_dataset, tokenizer, data_collator

def compute_metrics(eval_pred):
    # Try to load metrics with error handling
    try:
        accuracy = load("accuracy")
        f1 = load("f1")
        precision = load("precision")
        recall = load("recall")
    except Exception as e:
        print(f"Error loading metrics from Hub: {e}")
        print("Falling back to local metric computation")
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        acc = accuracy_score(labels, predictions)
        f1_macro = f1_score(labels, predictions, average='macro')
        prec = precision_score(labels, predictions, average='macro')
        rec = recall_score(labels, predictions, average='macro')
        
        f1_classes = f1_score(labels, predictions, average=None)
        f1_non_bionlp = f1_classes[0]  # index 0 corresponds to 'Non_BioNLP'
        f1_bionlp = f1_classes[1]      # index 1 corresponds to 'BioNLP'
        
        return {
            'accuracy': acc,
            'f1': f1_macro,
            'precision': prec,
            'recall': rec,
            'f1_non_bionlp': f1_non_bionlp,
            'f1_bionlp': f1_bionlp
        }
    
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    acc = accuracy.compute(predictions=predictions, references=labels)
    f1_score = f1.compute(predictions=predictions, references=labels, average='macro')
    prec = precision.compute(predictions=predictions, references=labels, average='macro')
    rec = recall.compute(predictions=predictions, references=labels, average='macro')
    
    # Compute per-class metrics
    f1_classes = f1.compute(predictions=predictions, references=labels, average=None)
    f1_non_bionlp = f1_classes['f1'][0]  # index 0 corresponds to 'Non_BioNLP'
    f1_bionlp = f1_classes['f1'][1]      # index 1 corresponds to 'BioNLP'
    
    return {
        'accuracy': acc['accuracy'],
        'f1': f1_score['f1'],
        'precision': prec['precision'],
        'recall': rec['recall'],
        'f1_non_bionlp': f1_non_bionlp,
        'f1_bionlp': f1_bionlp
    }

def main(model_name, jsonl_file, output_dir, max_samples=None):
    # Load and preprocess data
    dataset, label2id = load_and_preprocess_data(jsonl_file, max_samples=max_samples)
    
    # Tokenize the dataset
    tokenized_dataset, tokenizer, data_collator = tokenize_dataset(dataset, model_name)
    
    # Id2label for binary classification
    id2label = {v: k for k, v in label2id.items()}
    
    # Split the dataset according to 'split' column
    train_dataset = tokenized_dataset.filter(lambda example: example['split'] == 'train')
    eval_dataset = tokenized_dataset.filter(lambda example: example['split'] == 'dev')
    
    # Remove unnecessary columns
    train_dataset = train_dataset.remove_columns(['text', 'split'])
    eval_dataset = eval_dataset.remove_columns(['text', 'split'])
    
    # Load model for binary classification
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2, 
        id2label=id2label, 
        label2id=label2id
    )
    
    # Check if CUDA is available and set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--jsonl_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None,  # New argument
                      help="Limit processing to N samples (for testing)")
    
    args = parser.parse_args()
    main(args.model_name, args.jsonl_file, args.output_dir, args.max_samples)

