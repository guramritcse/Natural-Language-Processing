import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datasets import load_dataset
from svm import SVM
import json

# Load the model
def load_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

# Preprocess the data
def preprocess(data):
    processed_data = []
    processed_labels = []
    for sentence in data:
        processed_label = []
        for tag in sentence["ner_tags"]:
            processed_label.append(1 if tag > 0 else 0)
        processed_data.append(sentence["tokens"])
        processed_labels.append(processed_label)
    return processed_data, processed_labels

# Main function
def main():
    # Load the CoNLL-2003 dataset

    random.seed(42)

    dataset = load_dataset("conll2003")

    test_data = dataset["test"]

    test_data, test_labels = preprocess(test_data)
    

    combined_data = list(zip(test_data, test_labels))
    random.shuffle(combined_data)
    subset = combined_data[193:200]
    test_data, test_labels = zip(*subset)

    # Parameters
    number_of_tags = 2
    index_to_tag = {0: "O", 1: "B/I"}

    # Define the model
    model = load_model("./results/model.pkl")

    # Open a JSON file to store the results
    with open("./results/chat_results.json", "a") as json_file:
        for sentence, labels in zip(test_data, test_labels):
            # Model prediction
            predicted_tags = model.predict(sentence)
            
            # If the lengths of the predicted and actual labels don't match, skip this instance
            if len(predicted_tags) != len(labels):
                continue

            # Take user input as a list of tags
            while True:
                try:
                    print("###############################################\n")
                    chat_labels = input(f"Sentence: {sentence}\n")
                    chat_labels = eval(chat_labels)  # Convert input string to list

                    # Check if input size matches with labels
                    if len(chat_labels) == len(labels):
                        # Save the entry in JSON
                        entry = {
                            "sentence": sentence,
                            "label": labels,
                            "predlabel": predicted_tags,
                            "chatlabel": chat_labels
                        }
                        json.dump(entry, json_file)
                        json_file.write("\n")  # Separate entries with a newline
                        break
                    else:
                        print("Input size doesn't match. Please try again.")
                except Exception as e:
                    print(f"Invalid input format. Error: {e}")
                    continue
    
if __name__ == "__main__":
    main()
