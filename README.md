# BioNLP_BERT

A repository for BioNLP (Biomedical Natural Language Processing) classification using BERT models from Hugging Face.

## Overview

This repository contains scripts for training, evaluating, and performing inference with BERT models specifically adapted for biomedical text classification tasks.

## Dataset

The training dataset can be downloaded from [this Google Drive link](https://drive.google.com/file/d/1j5qy6D_mt-_e0azkp2HkOqOFua5r4AgO/view?usp=sharing).

## Usage

### Training

Train a BERT model on the BioNLP classification task:

```bash
python train.py \
  --model_name microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
  --jsonl_file all_data_classied_gpt4o.jsonl \
  --output_dir pubmedbert_bionlp_classification
```

#### Training Results

I've used a synthesized dataset with GPT-4o for PubMedBERT model training tesing with an 80/20 random split:

```
{
  'eval_loss': 0.0726708173751831,
  'eval_accuracy': 0.9863533757438949,
  'eval_f1': 0.9689875271633501,
  'eval_precision': 0.9694792457529896,
  'eval_recall': 0.9684972539469971,
  'eval_f1_non_bionlp': 0.9921943776043195,
  'eval_f1_bionlp': 0.9457806767223808,
  'eval_runtime': 93.952,
  'eval_samples_per_second': 103.734,
  'eval_steps_per_second': 1.628,
  'epoch': 4.0
}
```

### Evaluation

Evaluate a trained model on a test dataset:

```bash
python eval.py \
  --model_path ./pubmedbert_bionlp_classification/checkpoint-2440 \
  --jsonl_file 500_data_classied_gpt4o-mini_annotated_by_hk.jsonl
```

### Inference

Run inference using a trained model:

```bash
python inference.py \
  --model_path ./pubmedbert_bionlp_classification/checkpoint-2440 \
  --input_jsonl 500_data_classied_gpt4o-mini_annotated_by_hk.jsonl \
  --output_jsonl predictions.jsonl
```

## Supported Models

This repository works with biomedical BERT models available on Hugging Face, including:
- microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
- Other compatible BERT-based models
