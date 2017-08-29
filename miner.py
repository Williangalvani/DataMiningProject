import os
from tokenizer import extract_word_set
import json


# Find all .html files in "dataset" folder
for root, dirs, files in os.walk('dataset'):
    print(root, dirs, files)

    prep_dataset = root.replace("dataset", "prep_dataset")
    if not os.path.exists(prep_dataset):
        os.makedirs(prep_dataset)

    features = set()
    for file in files:
        if "html" in file:
            features = features | extract_word_set(os.path.join(root, file))


    with open(os.path.join(prep_dataset, "data.json"), 'w') as f:
        f.write(json.dumps(list(features)))