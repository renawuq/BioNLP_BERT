# BioNLP Paper Classification

## Overview
This project aims to classify papers as either BioNLP or Non_BioNLP based on their titles and abstracts. It involves manual annotation, validation of annotations, and the use of specific scripts for data processing and model training.

## Files and Directories
- `baseline.py`: The main script for baseline model.
- `train.py`: The main script to train and evaluate the binary classification model.
- `visual.ipynb`: Contains script to generated plot in my report.
- `requirements.txt`: A file listing required Python packages.
- `annotations/`: Directory containing JSONL files for training and evaluation.
  - `merged_annotations.jsonl`: Merged file containing annotations from both `annotated_50.jsonl` and `annotated_last30.jsonl`.
  - `big_training.jsonl`: Got from the [https://drive.google.com/file/d/1j5qy6D_mt-_e0azkp2HkOqOFua5r4AgO/view](https://drive.google.com/file/d/1j5qy6D_mt-_e0azkp2HkOqOFua5r4AgO/view). About 50,000 samples label by gpt. 
- `annotation_aggrement.ipynb`: Jupyter notebook detailing the data processing workflow and calculation of annotation agreement.
- `data_prepare.py`: Script used for manual annotation of papers.
- `papers.xlsx`: Raw data file containing the list of papers to be annotated and classified.
- `model_result`: A compress document of all pretrain model

**Noted many files mentioend above are not here due to the size, please check in following drive: https://drive.google.com/drive/folders/1gj2XKEQkRYTXCqllqvj90lCBQstsQkIv?usp=sharing" **. Looks like some file are way tooooo big, more than 2GB, so only part of the model get uploaded. Contact me for those model if you don't want to train by yourself, but I will delete them in 10 days .... 

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
please check requirements.txt, sometimes you need to check using chat-gpt to figure out, too

## Installation
Install the required packages using pip:

```bash
conda create -n bio_bert python=3.10.16 -y
conda activate bio_bert
pip install -r requirements.txt
```

#### baseline Model Results
``` bash
{
   accuracy: 0.7968
   f1: 0.7877
   precision: 0.6435
   recall: 0.8810
   f1_non_bionlp: 0.8317
   f1_bionlp: 0.7437
}
```


## Training the Model 1 (BERT-transformer, SLOW)
To train and evaluate the model using a pre-trained transformer, run the following command:

``` bash
python train.py --jsonl_file you_data_file --output_dir output --max_samples sample_size_you_want
## Example
python train.py --jsonl_file data_preapre/annotations/big_training.jsonl --output_dir train_samplesize_1000 --max_samples 1000
``` 

Arguments

--model_name: The name of the pre-trained transformer model (default: bert-base-uncased).

--jsonl_file: The path to the JSONL file containing the dataset.

--output_dir: The output directory where the model and checkpoints will be saved.

--max_samples: max samples you wanted for training 

## To Test the Model 1 (BERT-transformer, SLOW)
``` bash
python eval.py path_to_your_model jsonl_file_for_test_data
## Example:
python eval.py train_sample_100 data_preapre/annotations/test_data.jsonl 
```


## Training the Model 2 (Enhance, Hybrid-model)

``` bash
python model2.py you_data_file --output_dir output --max_samples sample_size_you_want
## Example
python way3_enhance.py data_preapre/annotations/big_training.jsonl --output_dir enhance_sample_100 --max_samples 100 
``` 

Arguments

--model_name: The name of the pre-trained transformer model (default: bert-base-uncased).

--jsonl_file: The path to the JSONL file containing the dataset.

--output_dir: The output directory where the model and checkpoints will be saved.

--max_samples: max samples you wanted for training 

## To Test the Model 2 (Enhance, Hybrid-model)
``` bash
python eval.py path_to_your_model jsonl_file_for_test_data
## Example:
python model2_eval.py enhance_sample_100 data_preapre/annotations/test_data.jsonl ```
``` 

## Following are the evaluation result using the 80 papers I annotated 
### Baseline model Evaluation Result
``` bash
{
   accuracy: 0.7968
   f1: 0.7877
   precision: 0.6435
   recall: 0.8810
   f1_non_bionlp: 0.8317
   f1_bionlp: 0.7437
}
```

### Model 1 Evaluation Result
``` bash
{
   accuracy: 0.7245
   f1: 0.6685
   precision: 0.7371
   recall: 0.6646
   f1_non_bionlp: 0.8047
   f1_bionlp: 0.5323
}
```

### Model 2 Evaluation Result
``` bash
{
   accuracy: 0.8575
   f1: 0.8508
   precision: 0.8477
   recall: 0.8551
   f1_non_bionlp: 0.8824
   f1_bionlp: 0.8193
}
```