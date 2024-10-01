import sys
import os
import pickle
import json
from tqdm import tqdm  # Import tqdm for progress tracking
from crf import CRF
import numpy as np

# Add the parent directory (HMM-Task) to sys.path
hmm_task_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'HMM-Task'))
sys.path.append(hmm_task_dir)
from hmm3 import HMM3

# Load the CRF and HMM models
def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Compare two lists of tags and return mismatches
def compare_tags(actual, predicted):
    return [(i, a, p) for i, (a, p) in enumerate(zip(actual, predicted), start=1) if a != p]

# Append results to a JSON file without overwriting

def append_to_json_file(filepath, new_data):
    # Load existing data or initialize an empty dictionary
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}

    # Update with new data
    data.update(new_data)

    # Write updated data back to the file
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

# Main function for prediction and mismatch categorization
def main():
    
    # unpickle models/test_data.pkl
    test_data = pickle.load(open("models/test_data.pkl", "rb"))
    # Load CRF and HMM models
    crf_model = load_model('models/crf_model.pkl')
    hmm_model = load_model('models/hmm_model.pkl')
    # Categorize the mismatches for HMM and CRF
    mismatch_categories = [0, 1, 2, 3, 4, 5, '>=6']
    sentences_processed = 0
    hmmCategory = None
    crfCategory = None
    for index, sentence in enumerate(test_data[:300]):
        sentences_processed += 1
        output_json = {}    
        joined_sentence = " ".join([word for word, tag in sentence])
        
        # Predict tags using CRF and HMM models
        hmmTags = hmm_model.predict([word for word, tag in sentence])
        crfTags = crf_model.predict([[word, "X"] for word, tag in sentence])
        actualTags = [tag for word, tag in sentence]
        words = [word for word, tag in sentence]

        assert len(hmmTags) == len(crfTags) == len(actualTags) == len(words)

        # Compare the actual tags with the predicted tags
        hmmMismatches = compare_tags(actualTags, hmmTags)
        crfMismatches = compare_tags(actualTags, crfTags)
        numHMM = len(hmmMismatches)
        numCRF = len(crfMismatches)
        hmmCategory = mismatch_categories[min(numHMM, 6)]
        if hmmCategory == 6:
            hmmCategory = '>=6'
        crfCategory = mismatch_categories[min(numCRF, 6)]
        if crfCategory == 6:
            crfCategory = '>=6'

        dir = f"data/HMM-Task_{hmmCategory}_wrong"
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        mismatch_tuples = []
        for word, actual, hmm, crf in zip(words, actualTags, hmmTags, crfTags):
            if crf != actual:
                mismatch_tuples.append([word, actual, hmm, crf])
            elif crf != hmm and hmm == actual:
                mismatch_tuples.append([word, actual, hmm, crf])
            elif crf != actual and hmm != actual:
                mismatch_tuples.append([word, actual, hmm, crf])
            elif hmm != actual:
                mismatch_tuples.append([word, actual, hmm, crf])
        
        output_json[index] = {
            "sentence": joined_sentence,
            "actual_tags": " ".join(actualTags),
            "hmm_tags": " ".join(hmmTags),
            "crf_tags": " ".join(crfTags),
            "hmm_mismatches": numHMM,
            "crf_mismatches": numCRF,
            "(word, actual, hmm, crf)": mismatch_tuples
        }

        append_to_json_file(f"{dir}/{crfCategory}_wrong.json", output_json)

        with open(f"{dir}/{crfCategory}_wrong.txt", 'a') as f:
            f.write(f"{index}: {joined_sentence}\n")

        with open('./sentences.txt', 'a') as f:
            f.write(f"{index}: {joined_sentence}\n")

    print(f"Processed {sentences_processed} sentences")

# main
if __name__ == "__main__":
    main()