import json
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertModel
from sklearn.metrics import classification_report
import joblib
import argparse
from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample

# Configuration
CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'max_features': 600,
    'ngram_range': (1, 3),
    'tfidf_params': {
        'stop_words': 'english',
        'lowercase': True,
        'analyzer': 'word',
        'max_df': 0.8,
        'min_df': 3
    },
    'bert_params': {
        'model_name': 'distilbert-base-uncased',
        'max_length': 128,
        'batch_size': 32,  # Increased for better GPU utilization
        'num_layers': 4  # Last N layers to use
    },
    'output_dir': 'enhanced_model'
}

# Domain knowledge
BIOMED_TERMS = ['clinical', 'drug', 'protein', 'genomic', 'medical', 'pharma', 'therapy']
NLP_TERMS = ['ner', 'relation extraction', 'text mining', 'embedding', 'tokenization', 'named entity']

def load_data(filepath, max_samples=None):
    """Load and preprocess JSONL data with sampling support"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading data"):
            try:
                item = json.loads(line)
                
                # Check for required fields
                if not ('title' in item and 'abstract' in item):
                    continue
                    
                # Handle both label formats
                if 'label' in item and item['label'] in [0, 1]:
                    label = item['label']
                elif 'is_bionlp' in item and item['is_bionlp'] in ['BioNLP', 'Non_BioNLP']:
                    label = 1 if item['is_bionlp'] == 'BioNLP' else 0
                else:
                    continue
                
                data.append({
                    'title': item['title'],
                    'abstract': item['abstract'],
                    'text': f"Title: {item['title']}\nAbstract: {item['abstract']}",
                    'label': label
                })
            except json.JSONDecodeError:
                continue
    
    if not data:
        raise ValueError("No valid data found in the input file")
    
    df = pd.DataFrame(data)
    
    # Stratified sampling if max_samples specified
    if max_samples and len(df) > max_samples:
        df = resample(
            df,
            replace=False,
            n_samples=max_samples,
            random_state=CONFIG['random_state'],
            stratify=df['label']
        )
    
    return df

def extract_domain_features(texts):
    """Create domain-specific manual features"""
    features = []
    for text in texts:
        text_lower = text.lower()
        biomed_score = sum(text_lower.count(term) for term in BIOMED_TERMS)
        nlp_score = sum(text_lower.count(term) for term in NLP_TERMS)
        has_sections = int('method:' in text_lower or 'results:' in text_lower)
        ref_count = text_lower.count('reference')
        features.append([biomed_score, nlp_score, has_sections, ref_count])
    return np.array(features)

def create_hybrid_features(df, tokenizer, bert_model, device, fit_vectorizer=True, tfidf=None):
    """Enhanced feature engineering pipeline"""
    # 1. TF-IDF features
    if fit_vectorizer:
        tfidf = TfidfVectorizer(
            max_features=CONFIG['max_features'],
            ngram_range=CONFIG['ngram_range'],
            **CONFIG['tfidf_params']
        )
        tfidf_feats = tfidf.fit_transform(df['text'])
    else:
        tfidf_feats = tfidf.transform(df['text'])
    
    # 2. BERT features (batched processing)
    bert_feats = []
    bert_model.eval()
    texts = df['text'].tolist()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), CONFIG['bert_params']['batch_size']), 
                     desc="Extracting BERT embeddings"):
            batch = texts[i:i+CONFIG['bert_params']['batch_size']]
            inputs = tokenizer(
                batch,
                truncation=True,
                padding='max_length',
                max_length=CONFIG['bert_params']['max_length'],
                return_tensors='pt'
            ).to(device)
            
            outputs = bert_model(**inputs)
            # Weighted average of last N layers
            layer_weights = [0.2, 0.25, 0.3, 0.25]
            weighted_embeddings = torch.zeros_like(outputs.last_hidden_state[:, 0, :])
            for i, weight in enumerate(layer_weights):
                weighted_embeddings += weight * outputs.hidden_states[-(i+1)][:, 0, :]
            bert_feats.extend(weighted_embeddings.cpu().numpy())
    
    # 3. Domain features
    domain_feats = extract_domain_features(df['text'])
    
    # Combine all features
    return np.hstack([
        tfidf_feats.toarray(),
        np.array(bert_feats),
        domain_feats
    ]), tfidf

def train_model(X_train, y_train, X_test, y_test):
    """Train with enhanced parameters"""
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights[1] *= 1.5  # Boost minority class
    
    model = GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=CONFIG['random_state']
    )
    
    model.fit(X_train, y_train)
    
    print("\nTraining Set Performance:")
    print(classification_report(y_train, model.predict(X_train), 
          target_names=['Non_BioNLP', 'BioNLP']))
    
    print("\nTest Set Performance:")
    print(classification_report(y_test, model.predict(X_test), 
          target_names=['Non_BioNLP', 'BioNLP']))
    
    return model

def main(input_file, output_dir, max_samples=None):
    """Main execution with sampling support"""
    print("Loading data...")
    df = load_data(input_file, max_samples=max_samples)
    
    # Split data
    train_df, test_df = train_test_split(
        df,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'],
        stratify=df['label']
    )
    
    # Initialize BERT
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = DistilBertConfig.from_pretrained(
        CONFIG['bert_params']['model_name'],
        output_hidden_states=True
    )
    tokenizer = DistilBertTokenizer.from_pretrained(CONFIG['bert_params']['model_name'])
    bert_model = DistilBertModel.from_pretrained(
        CONFIG['bert_params']['model_name'],
        config=config
    ).to(device)
    
    # Create features
    print("\nCreating features...")
    X_train, tfidf = create_hybrid_features(train_df, tokenizer, bert_model, device)
    X_test, _ = create_hybrid_features(test_df, tokenizer, bert_model, device, False, tfidf)
    y_train, y_test = train_df['label'].values, test_df['label'].values
    
    # Train
    print("\nTraining model...")
    model = train_model(X_train, y_train, X_test, y_test)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump({
        'model': model,
        'tfidf': tfidf,
        'feature_dim': X_train.shape[1],
        'class_names': ['Non_BioNLP', 'BioNLP']
    }, os.path.join(output_dir, 'model.joblib'))
    
    bert_model.save_pretrained(os.path.join(output_dir, 'bert_model'))
    tokenizer.save_pretrained(os.path.join(output_dir, 'bert_tokenizer'))
    
    print(f"\nModel saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help="Path to JSONL training data")
    parser.add_argument('--output_dir', default=CONFIG['output_dir'], help="Output directory")
    parser.add_argument('--max_samples', type=int, default=None,
                      help="Maximum number of samples to use (randomly selected with class balance)")
    args = parser.parse_args()
    
    main(args.input_file, args.output_dir, args.max_samples)