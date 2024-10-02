import sys
import os
import pickle
import json
from tqdm import tqdm  # Import tqdm for progress tracking
from crf import CRF
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory (HMM-Task) to sys.path
hmm_task_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'HMM-Task'))
sys.path.append(hmm_task_dir)
from hmm3 import HMM3

tag_to_index = {'ADJ': 0, 'ADP': 1, 'ADV': 2, 'CONJ': 3, 'DET': 4, 'NOUN': 5, 'NUM': 6, 'PRON': 7, 'PRT': 8, 'VERB': 9, '.': 10, 'X': 11}
index_to_tag = {0: 'ADJ', 1: 'ADP', 2: 'ADV', 3: 'CONJ', 4: 'DET', 5: 'NOUN', 6: 'NUM', 7: 'PRON', 8: 'PRT', 9: 'VERB', 10: '.', 11: 'X'}

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
    # create a 2D matrix with 12 rows, 12 columns, names being the elements of the tag_to_index dictionary
    confusion_matrix_crf = np.zeros((len(tag_to_index), len(tag_to_index)), dtype=int)
    confusion_matrix_hmm = np.zeros((len(tag_to_index), len(tag_to_index)), dtype=int)

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
    mismatches = 0
    for index, sentence in enumerate(test_data[:400]):
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

        dir = f"datasets/HMM-Task_{hmmCategory}_wrong"
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        
        mismatch_tuples = []
        for word, actual, hmm, crf in zip(words, actualTags, hmmTags, crfTags):
            if crf != actual or hmm != actual:
                mismatch_tuples.append([word, actual, hmm, crf])
                
            confusion_matrix_crf[tag_to_index[crf]][tag_to_index[actual]] += 1
        
            confusion_matrix_hmm[tag_to_index[hmm]][tag_to_index[actual]] += 1
            # Remove the lines that decrement the values in the confusion matrices

            if actual == 'NOUN' and crf != 'NOUN':
                mismatches += 1
                print(f"Actual: {actual}, CRF: {crf}, Sentence: {joined_sentence}, Word: {word}")
            # if actual == 'ADJ' and crf != 'ADJ':
            #     mismatches += 1
            #     print(f"Actual: {actual}, CRF: {crf}, Sentence: {joined_sentence}, Word: {word}")
            # if actual == 'ADP' and crf != 'ADP':
            #     mismatches += 1
            #     print(f"Actual: {actual}, CRF: {crf}, Sentence: {joined_sentence}, Word: {word}")
            # if actual == 'NOUN' and crf != 'NOUN':
            #     mismatches += 1
            #     print(f"Actual: {actual}, CRF: {crf}, Sentence: {joined_sentence}, Word: {word}")
                
            
        
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

    print(mismatches)
    print(f"Processed {sentences_processed} sentences")

 # Normalize the confusion matrices
 # round off the values to 2 decimal places

    confusion_matrix_crf_normalized = np.round(confusion_matrix_crf / np.sum(confusion_matrix_crf, axis=1, keepdims=True), 2)
    confusion_matrix_hmm_normalized = np.round(confusion_matrix_hmm / np.sum(confusion_matrix_hmm, axis=1, keepdims=True), 2)

    # Plot the confusion matrix as heatmap side by side
    plt.figure(figsize=(12, 6))
    
    # CRF Model Confusion Matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(confusion_matrix_crf_normalized, fmt='.2f', annot=True, annot_kws={"size": 8}, cmap='viridis',xticklabels=index_to_tag.values(), yticklabels=index_to_tag.values(), square=True)
    
    plt.title('CRF Model')
    plt.xlabel('Actual')  # Change to 'Actual' for consistency
    plt.ylabel('Predicted')  # Change to 'Predicted' for consistency
    
    # HMM Model Confusion Matrix
    plt.subplot(1, 2, 2)
    sns.heatmap(confusion_matrix_hmm_normalized, fmt='.2f', annot=True, annot_kws={"size": 8}, cmap='viridis',xticklabels=index_to_tag.values(), yticklabels=index_to_tag.values(), square=True)

    plt.title('HMM Model')
    plt.xlabel('Actual')  # Change to 'Actual' for consistency
    plt.ylabel('Predicted')  # Change to 'Predicted' for consistency
    
    plt.tight_layout()
    plt.savefig(f'{dir}/confusion_matrix.png')




# main
if __name__ == "__main__":
    main()