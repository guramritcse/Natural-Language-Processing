import nltk
from nltk.corpus import brown
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from crf import CRF
import json
import re
import os

# Load the model
def load_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def compare_lists(list1, list2):
    mismatches = [(i, x, y) for i, (x, y) in enumerate(zip(list1, list2), start=1) if x != y]
    return mismatches

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

# Main function
def main():
    # Load the model
    model = load_model("results/model.pkl")

    # pick indices from ../HMM-Task/1_wrong.json
    for filename in os.listdir("../HMM-Task/datasets"):
        if filename.endswith("_wrong.json"):
            with open(f"../HMM-Task/datasets/{filename}", 'r') as f:
                file_number = filename.split("_")[0]
                data = json.load(f)
                
                for key, value in data.items():
                    output_json = {}
                    sentence = value["words"]
                    words = sentence.split()

                    tags_list = value["actual_tags"].split()

                    # Combine them into a list of tuples
                    word_tag_pairs = [[word, "X"] for word in words]

                    crf_tags = model.predict(word_tag_pairs)
                    hmm_tags = value["predicted_tags"].split()
                    actual_tags = value["actual_tags"].split()
                    mismatches = compare_lists(actual_tags, crf_tags)
                    num_mismatches = len(mismatches)
                    # assert len(actual_tags) == len(crf_tags) == len(hmm_tags)
                    assert len(actual_tags) == len(crf_tags) == len(hmm_tags), \
                        f"Tag length mismatch! Actual: {len(actual_tags)}, CRF: {len(crf_tags)}, HMM: {len(hmm_tags)}"
                    # Collect and filter mismatches based on specified conditions

                    mismatch_tuples = []
                    for word, actual, crf, hmm in zip(words, actual_tags, crf_tags, hmm_tags):
                        # CRF and actual differ
                        if crf != actual:
                            mismatch_tuples.append((word, actual, crf, hmm))
                        # CRF and HMM differ but HMM is the same as actual
                        elif crf != hmm and hmm == actual:
                            mismatch_tuples.append((word, actual, crf, hmm))
                        # Both CRF and HMM differ from actual
                        elif crf != actual and hmm != actual:
                            mismatch_tuples.append((word, actual, crf, hmm))
                        # HMM differs from actual
                        elif hmm != actual:
                            mismatch_tuples.append((word, actual, crf, hmm))


                    if file_number == ">=6":
                        
                        output_json[key] = {
                                "words": sentence,
                                "actual_tags": value["actual_tags"],
                                "CRF_tags": " ".join(crf_tags),
                                "num_CRF_mismatches": num_mismatches,
                                "HMM_tags": value["predicted_tags"],
                                "num_HMM_mismatches": value["num_wrong_tags"],
                                "(Word, Actual, CRF, HMM)": mismatch_tuples
                            }
                    else:
                        output_json[key] = {
                        "words": sentence,
                        "actual_tags": value["actual_tags"],
                        "CRF_tags": " ".join(crf_tags),
                        "HMM_tags": value["predicted_tags"],
                        "(Word, Actual, CRF, HMM)": mismatch_tuples
                    }
                        
                    file_prefix = filename.split(".")[0]
                    if 0 <= num_mismatches < 6:
                        file_prefix = filename.split(".")[0]
                        if not os.path.exists(f"datasets/HMM-Task_{file_prefix}"):
                            os.makedirs(f"datasets/HMM-Task_{file_prefix}")

                        # Write to text file
                        with open(f"datasets/HMM-Task_{file_prefix}/{num_mismatches}_wrong.txt", 'a') as f:
                            f.write(f"{key}: {sentence}\n")

                        # Append to the JSON file (ensuring we do not overwrite)
                        append_to_json_file(f"datasets/HMM-Task_{file_prefix}/{num_mismatches}_wrong.json", output_json)

                    # Save JSON and text files for >= 6 mismatches
                    elif num_mismatches >= 6:
                        file_prefix = filename.split(".")[0]
                        if not os.path.exists(f"datasets/HMM-Task_{file_prefix}"):
                            os.makedirs(f"datasets/HMM-Task_{file_prefix}")

                        # Write to text file
                        with open(f"datasets/HMM-Task_{file_prefix}/>=6_wrong.txt", 'a') as f:
                            f.write(f"{key}: {sentence}\n")

                        # Append to the JSON file (ensuring we do not overwrite)
                        append_to_json_file(f"datasets/HMM-Task_{file_prefix}/>=6_wrong.json", output_json)
                

if __name__ == "__main__":
    main()