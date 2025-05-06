# Project Title: Biomedical Paper Classification

## Overview
This project aims to classify papers as either BioNLP or Non_BioNLP based on their titles and abstracts. It involves manual annotation, validation of annotations, and the use of specific scripts for data processing and model training.

## Files and Directories
- `train.py`: The main script to train and evaluate the binary classification model.
- `requirements.txt`: A file listing required Python packages.
- `annotations/`: Directory containing JSONL files for training and evaluation.
  - `merged_annotations.jsonl`: Merged file containing annotations from both `annotated_50.jsonl` and `annotated_last30.jsonl`.
- `annotation_aggrement.ipynb`: Jupyter notebook detailing the data processing workflow and calculation of annotation agreement.
- `data_prepare.py`: Script used for manual annotation of papers.
- `papers.xlsx`: Raw data file containing the list of papers to be annotated and classified.

## Data Processing Workflow
Details on the data processing steps are provided in the `annotation_aggrement.ipynb` notebook, including:
1. **Manual Annotation**:
   - Annotate the first 50 papers and the last 30 papers in `papers.xlsx` using the `data_prepare.py` script.
   - Validate these annotations with annotations from HK and QC, resulting in a Cohen's Kappa score of 0.87.
   - The validated annotations are saved in `annotations/annotated_50.jsonl` and `annotations/annotated_last30.jsonl`.

2. **Merging Annotations**:
   - The annotated files are merged into `annotations/merged_annotations.jsonl` to prepare the training dataset.

3. **Test Dataset Preparation**:
   - Papers from row 52 to row 302 in `papers.xlsx` are used as the test dataset called "test_data.jsonl".

## Dependencies
- Python 3.7+
- pandas
- scikit-learn
- datasets
- transformers
- evaluate
- torch

## Installation
Install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Training the Model
To train and evaluate the model using a pre-trained transformer, run the following command:

``` bash
python train.py --model_name bert-base-uncased --jsonl_file data_preapre/annotations/merged_annotations.jsonl --output_dir output/
``` 

Arguments
--model_name: The name of the pre-trained transformer model (default: bert-base-uncased).
--jsonl_file: The path to the JSONL file containing the dataset.
--output_dir: The output directory where the model and checkpoints will be saved.

#### baseline Model Training Results

I've used a synthesized dataset with GPT-4o for PubMedBERT model training tesing with an 80/20 random split:

```
{
   'eval_loss': 0.20169168710708618, 
   'eval_accuracy': 0.9375, 
   'eval_f1': 0.9372549019607843, 
   'eval_precision': 0.9375, 
   'eval_recall': 0.9444444444444444, 
   'eval_f1_non_bionlp': 0.9411764705882353, 
   'eval_f1_bionlp': 0.9333333333333333, 
   'eval_runtime': 4.6928, 
   'eval_samples_per_second': 3.409, 
   'eval_steps_per_second': 0.852, 
   'epoch': 5.0
}
```


### Baseline model Evaluation
Evaluate a trained model on a test dataset:
``` bash
python eval.py --model_path ./output/checkpoint-80 --jsonl_file data_preapre/annotations/test_data.jsonl
``` 
Arguments
--model_name: The name of the model path you want to use 
--jsonl_file: The path to the JSONL file containing the dataset.

### Baseline model Evaluation Result
```
{
   accuracy: 0.7791
   f1: 0.7771
   precision: 0.7850
   recall: 0.8011
   f1_non_bionlp: 0.7983
   f1_bionlp: 0.7559
}
```
