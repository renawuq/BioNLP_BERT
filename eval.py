import json
import pandas as pd
from datasets import Dataset
import torch
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import evaluate
from tqdm import tqdm

def load_and_preprocess_data(jsonl_file):
    # Load the dataset from JSONL file
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    # Filter for required fields and valid labels
    filtered_data = []
    for item in data:
        # if all(key in item for key in ['title', 'abstract', 'is_bionlp']) and item['is_bionlp'] in ['BioNLP', 'Non_BioNLP']:
        #    filtered_data.append(item)
        if all(key in item for key in ['title', 'abstract', 'label']) and item['label'] in [0, 1]:
            # Convert to expected format for compatibility
            item['is_bionlp'] = 'BioNLP' if item['label'] == 1 else 'Non_BioNLP'
            filtered_data.append(item)


    df = pd.DataFrame(filtered_data)
    df['text'] = 'Title: ' + df['title'].fillna('') + '\nAbstract: ' + df['abstract'].fillna('')
    label2id = {'BioNLP': 1, 'Non_BioNLP': 0}
    df['label'] = df['is_bionlp'].map(label2id)

    df = df[['text', 'label']]  
    dataset = Dataset.from_pandas(df)
    return dataset, label2id


def compute_metrics(predictions, labels):
    acc = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    prec_macro = precision_score(labels, predictions, average='macro')
    rec_macro = recall_score(labels, predictions, average='macro')
    
    # Get per-class F1 scores
    f1_classes = f1_score(labels, predictions, average=None)
    
    # Handle binary or multi-class cases
    if len(f1_classes) >= 2:  # Binary case
        f1_non_bionlp = f1_classes[0]
        f1_bionlp = f1_classes[1]
    else:  # Handle unexpected cases
        f1_non_bionlp = float('nan')
        f1_bionlp = float('nan')
    
    return {
        'accuracy': acc,
        'f1': f1_macro,
        'precision': prec_macro,
        'recall': rec_macro,
        'f1_non_bionlp': f1_non_bionlp,
        'f1_bionlp': f1_bionlp
    }

def evaluate_model(model_path, eval_dataset, batch_size=16):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    model.to(device)

    def preprocess_function(examples, tokenizer):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    tokenized_eval_dataset = eval_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    tokenized_eval_dataset = tokenized_eval_dataset.remove_columns(
        [col for col in tokenized_eval_dataset.column_names if col not in ['input_ids', 'attention_mask', 'label']]
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    eval_dataloader = torch.utils.data.DataLoader(tokenized_eval_dataset, batch_size=batch_size, collate_fn=data_collator)

    model.eval()
    predictions = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**{k: v for k, v in batch.items() if k != 'label'})
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            labels.extend(batch['labels'].cpu().numpy())
            
    return compute_metrics(predictions, labels)


def main(model_path, jsonl_file):
    # Load and preprocess data
    eval_dataset, _ = load_and_preprocess_data(jsonl_file)
    
    # Evaluate the model
    metrics = evaluate_model(model_path, eval_dataset)
    
    # Print metrics
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a BioNLP classification model.")
    parser.add_argument("--model_path", type=str, required=True, 
                      help="Path to the trained model directory")
    parser.add_argument("--jsonl_file", type=str, required=True, 
                      help="The JSONL file containing the dataset")
    
    args = parser.parse_args()
    main(args.model_path, args.jsonl_file)
