import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, classification_report
)
from sklearn.model_selection import train_test_split

# Set random seed
np.random.seed(42)

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def extract_text_and_labels(records):
    texts, labels = [], []
    for r in records:
        if 'abstract' in r and 'is_bionlp' in r:
            texts.append((r.get('title', '') + ' ' + (r.get('abstract', '')).strip()))
            labels.append(1 if r['is_bionlp'] == 'BioNLP' else 0)
        elif 'text' in r and 'label' in r:
            texts.append(r['text'])
            labels.append(r['label'])
        elif 'title' in r and 'abstract' in r and 'label' in r:
            texts.append(f"Title: {r['title']}\nAbstract: {r['abstract']}")
            labels.append(r['label'])
    return texts, labels

def run_tfidf_baseline(train_records, test_records):
    train_texts, train_labels = extract_text_and_labels(train_records)
    test_texts, test_labels = extract_text_and_labels(test_records)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    clf = LogisticRegression(class_weight='balanced', max_iter=1000)
    clf.fit(X_train, train_labels)
    preds = clf.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(test_labels, preds),
        'f1': f1_score(test_labels, preds, average='macro'),
        'precision': precision_score(test_labels, preds),
        'recall': recall_score(test_labels, preds),
        'f1_non_bionlp': f1_score(test_labels, preds, pos_label=0),
        'f1_bionlp': f1_score(test_labels, preds, pos_label=1),
    }

    print("\n=== TF-IDF Results ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.bar(metrics.keys(), metrics.values(), color=['steelblue'] * 4 + ['green', 'darkorange'])
    plt.title("TF-IDF Performance Metrics")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    return metrics

def prepare_datasets(jsonl_path):
    raw_data = load_jsonl(jsonl_path)
    # Use first 50 + last 30 as manual train, 52–302 as test
    train_records = raw_data[:50] + raw_data[-30:]
    test_records = raw_data[51:302]
    return train_records, test_records

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_file", type=str, required=True)
    args = parser.parse_args()

    train_data, test_data = prepare_datasets(args.jsonl_file)
    run_tfidf_baseline(train_data, test_data)
