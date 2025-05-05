# BioNLP Paper Classification: Manual Annotation vs. GPT-Labeled Baseline

This project explores the task of classifying biomedical research abstracts as **BioNLP-related** or **not**, using BERT-based models. We compare a strong baseline model trained on GPT-annotated data with our own model trained on **50 manually annotated samples**.

---

## Overview

Biomedical Natural Language Processing (BioNLP) focuses on applying NLP to biomedical texts. Identifying whether a research paper falls within this domain is crucial for downstream tasks like literature triage or domain-specific model training.

In this project:
- We manually annotated a small sample (50 papers).
- We trained a model on this set using `Bio_ClinicalBERT`.
- We compared it to a baseline trained on GPT-labeled data (provided by the course team).

---

## Dataset

### Baseline Dataset
- **Filename**: `all_data_classified_gpt4o.jsonl`
- **Source**: Provided via the [BioNLP_BERT GitHub repository](https://github.com/YinYuBB/BioNLP_BERT)
- **Labels**: Annotated using GPT-4o with ~500 examples
- **Fields**:
  - `title`
  - `abstract`
  - `label` (`1` for BioNLP-related, `0` otherwise)

### Manual Dataset (Ours)
- **Filename**: `manual_annotated_50.jsonl`
- **Annotations**: Done manually by our team following BioNLP relevance guidelines.
- **Labels**:
  - `1` if the paper relates to BioNLP (e.g., entity linking, parsing, clinical text mining)
  - `0` otherwise
- **Imbalance**: The 50-paper set is highly imbalanced (e.g., ~40 negatives and ~10 positives).

---

## Methodology

### 1. Data Preprocessing
- Combine `title` and `abstract` into a single input string.
- Tokenize using `emilyalsentzer/Bio_ClinicalBERT`.

### 2. Manual Annotation
- We labeled the **first 50 abstracts manually**.
- Disagreements and borderline cases were resolved through team discussion.
- The goal was to see how far we could go with **small but high-quality human-labeled data**.
#### Annotation Process

To ensure high-quality labels, we manually annotated the first 50 biomedical papers using a custom Python tool:

#### `annotate.py`

This script provides an interface for annotators to:

- Load abstracts and titles from the raw dataset
- Display each paper one by one
- Assign a binary label (`1` for BioNLP-related, `0` otherwise)
- Save annotations to a new JSONL file (`manual_annotated_50.jsonl`)

The manual annotation followed strict guidelines:
- Only label a paper as BioNLP if it explicitly involves natural language processing applied to biomedical or clinical texts.
- Use external resources (PubMed, Wikipedia) when the paper’s scope is ambiguous.
- When uncertain, annotators consulted with each other to resolve disagreements.

This careful process ensures that our small dataset, while limited in size, has **high-quality, human-verified labels**.

---



### 3. Addressing Class Imbalance
- We tried two approaches:
  - **Undersampling** the majority class
  - **Class-weighted binary cross-entropy loss** (preferred)
- These methods prevent the model from being biased toward predicting only the majority class.

### 4. Model Training
- Model: `Bio_ClinicalBERT` from Hugging Face Transformers
- Loss: Weighted `BCEWithLogitsLoss`
- Training Epochs: 3–5
- Batch Size: 16
- Split: 80/20 train-validation split

### 5. Evaluation
- Accuracy, Precision, Recall, and F1 Score
- Emphasis on **F1 for minority class (BioNLP = 1)**

---

## 📊 Results

We trained the baseline `Bio_ClinicalBERT` model on our manually annotated set of 60 abstracts. Below are the evaluation results per epoch on the validation split (separate 20 samples: unseen data).

| Metric          | Value   |
|-----------------|---------|
| Accuracy        | 0.7500  |
| F1 Score        | 0.7460  |
| Precision       | 0.7667  |
| Recall          | 0.7500  |
| F1 (Non-BioNLP) | 0.7143  |
| F1 (BioNLP)     | 0.7778  |

Based on the inference.py and use the unseen_data.jsonl, we got: 
Summary:
  BioNLP papers: 44 (88.0%)
  Non-BioNLP papers: 6 (12.0%)




## 🔗 Baseline Reference

This repository uses the training pipeline and scripts provided by the course team:

🔗 [BioNLP_BERT GitHub](https://github.com/YinYuBB/BioNLP_BERT)

Example training command from the baseline repo:

```bash
python train.py \
  --model_name microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
  --jsonl_file all_data_classied_gpt4o.jsonl \
  --output_dir pubmedbert_bionlp_classification

