import json
from sklearn.model_selection import train_test_split

# Load the annotated files
file_1 = "annotations/annotated_50.jsonl"
file_2 = "annotations/annotated_last30.jsonl"

def load_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

# Load data
data_1 = load_jsonl(file_1)
data_2 = load_jsonl(file_2)

# Combine data
combined_data = data_1 + data_2

# Count label distribution
label_counts = {0: 0, 1: 0}
for item in combined_data:
    label_counts[item['label']] += 1

# Separate by label
label_0 = [item for item in combined_data if item['label'] == 0]
label_1 = [item for item in combined_data if item['label'] == 1]

# Make balanced set by undersampling the majority class
min_count = min(len(label_0), len(label_1))
balanced_data = label_0[:min_count] + label_1[:min_count]

# Shuffle and split into train and test (80/20)
train_data, test_data = train_test_split(balanced_data, test_size=0.2, random_state=42, stratify=[item['label'] for item in balanced_data])

# Save to new files
train_file = "annotations/balanced_train.jsonl"
test_file = "annotations/balanced_test.jsonl"

with open(train_file, "w") as f_train, open(test_file, "w") as f_test:
    for item in train_data:
        json.dump(item, f_train)
        f_train.write("\n")
    for item in test_data:
        json.dump(item, f_test)
        f_test.write("\n")

