import json
import re
import pandas as pd
import numpy as np
def load_and_preprocess_data(jsonl_file):
    """Load JSONL data, clean text, split into train/dev, and check balance."""
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line")

    filtered_data = [
        item for item in data 
        if all(key in item for key in ['title', 'abstract', 'label']) 
        and item['label'] in [0, 1]
    ]
    
    df = pd.DataFrame(filtered_data)
    # After filtering data, check class balance
    print("\n=== Class Distribution ===")
    print(f"Total samples: {len(df)}")
    print(df['label'].value_counts(normalize=True).rename({0: 'Non_BioNLP', 1: 'BioNLP'}))

    # Plot (optional)
    import matplotlib.pyplot as plt
    df['label'].value_counts().plot(kind='bar', title='Class Distribution')
    plt.xticks([0, 1], ['Non_BioNLP', 'BioNLP'], rotation=0)
    plt.show()

load_and_preprocess_data("data_preapre/annotations/merged_annotations.jsonl")