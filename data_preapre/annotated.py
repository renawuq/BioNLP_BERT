import json

INPUT_FILE = "bio_papers_raw_data.jsonl"
OUTPUT_FILE = "annotations/annotated_last30.jsonl"
NUM_ENTRIES = 30
annotated_data = []

# Read all lines first
with open(INPUT_FILE, "r") as infile:
    all_lines = infile.readlines()

# Take the last NUM_ENTRIES lines
selected_lines = all_lines[-NUM_ENTRIES:]
# Take first 
# selected_lines = all_lines[NUM_ENTRIES:]
# Annotate
for i, line in enumerate(selected_lines):
    paper = json.loads(line)
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")

    print(f"\n--- Paper {i + 1} ---")
    print(f"Title: {title}")
    print(f"Abstract: {abstract}\n")

    while True:
        label_input = input("Label this paper as BioNLP? (1 = Yes, 0 = No): ").strip()
        if label_input in {"0", "1"}:
            paper["label"] = int(label_input)
            annotated_data.append(paper)
            break
        else:
            print("Invalid input. Please enter 0 or 1.")

# Save to output file
with open(OUTPUT_FILE, "w") as outfile:
    for paper in annotated_data:
        json.dump(paper, outfile)
        outfile.write("\n")

print(f"\nAnnotation of last {NUM_ENTRIES} entries complete. Saved to {OUTPUT_FILE}")
