import json
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
import torch
import argparse
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(jsonl_file):
    # Load the dataset from JSONL file
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line")
    
    # Filter for required fields and valid labels
    filtered_data = []
    for item in data:
        if all(key in item for key in ['title', 'abstract', 'is_bionlp']) and item['is_bionlp'] in ['BioNLP', 'Non_BioNLP']:
            filtered_data.append(item)
    
    # Convert to DataFrame
    df = pd.DataFrame(filtered_data)
    
    # Combine title and abstract as input text
    df['text'] = 'Title: ' + df['title'].fillna('') + '\nAbstract: ' + df['abstract'].fillna('')
    
    # Define label mapping for binary classification
    label2id = {
        'BioNLP': 1,
        'Non_BioNLP': 0
    }
    
    # Encode the labels
    df['label'] = df['is_bionlp'].map(label2id)
    
    # Split the data into train and dev sets if 'split' column doesn't exist
    if 'split' not in df.columns:
        train_df, dev_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
        train_df['split'] = 'train'
        dev_df['split'] = 'dev'
        df = pd.concat([train_df, dev_df])
    
    # Clean the data by selecting only the necessary columns
    df = df[['text', 'label', 'split']]
    
    # Transform the pandas DataFrame to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    
    return dataset, label2id

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
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    
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

def main(model_name, jsonl_file, output_dir):
    # Load and preprocess data
    dataset, label2id = load_and_preprocess_data(jsonl_file)
    
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
        learning_rate=3e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=5,
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
    parser = argparse.ArgumentParser(description="Train a binary classification model for BioNLP.")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                      help="The name of the pre-trained model.")
    parser.add_argument("--jsonl_file", type=str, required=True,
                      help="The JSONL file containing the dataset.")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="The output directory where the model and checkpoints will be written.")
    
    args = parser.parse_args()
    main(args.model_name, args.jsonl_file, args.output_dir)


