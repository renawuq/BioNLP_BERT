import json
import os

# Config
INPUT_FILE = "data_preapre/annotations/bio_papers_raw_data.jsonl"
OUTPUT_FILE = "data_preapre/annotations/unseen_data.jsonl"
NUM_SKIP_FIRST = 50
NUM_SKIP_LAST = 30
NUM_SELECT = 50  # how many to select from unseen portion

# Load all entries
with open(INPUT_FILE, "r") as f:
    lines = f.readlines()

# Sanity check
total = len(lines)
unseen_start = NUM_SKIP_FIRST
unseen_end = total - NUM_SKIP_LAST
unseen_range = lines[unseen_start:unseen_end]

print(f"Total entries: {total}")
print(f"Selecting from index {unseen_start} to {unseen_end - 1} (unseen region)")

# Select up to NUM_SELECT
selected_unseen = unseen_range[:NUM_SELECT]

# Save output
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    for line in selected_unseen:
        f.write(line)

print(f"Saved {len(selected_unseen)} unseen entries to {OUTPUT_FILE}")
