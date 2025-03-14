import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datasets import load_dataset
from svm import SVM

# Save the model
def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

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

# Evaluate the model on the test data
def evaluate(models, test_data, test_labels, number_of_tags):
    correct_tokens = 0
    total_tokens = 0
    correct_sentences = 0
    total_sentences = 0
    confusion_matrix = np.zeros((number_of_tags, number_of_tags))
    predicted_tags_test_1 = models[0].predict(test_data, desc=1, type=1)
    predicted_tags_test_2 = models[1].predict(test_data, desc=1, type=2)
    for predicted_tags_1, predicted_tags_2, labels in zip(predicted_tags_test_1, predicted_tags_test_2, test_labels):
        if len(predicted_tags_1) != len(labels) or len(predicted_tags_2) != len(labels):
            print("Length mismatch")
            continue
        all_correct = True
        for predicted_tag_1, predicted_tag_2, actual_tag in zip(predicted_tags_1, predicted_tags_2, labels):
            predicted_tag = 1 if predicted_tag_1 == 1 or predicted_tag_2 == 1 else 0
            confusion_matrix[predicted_tag][actual_tag] += 1
            total_tokens += 1
            if predicted_tag == actual_tag:
                correct_tokens += 1
            else:
                all_correct = False
        if all_correct:
            correct_sentences += 1
        total_sentences += 1
    token_acc = correct_tokens / total_tokens
    sentence_acc = correct_sentences / total_sentences
    confusion_matrix = confusion_matrix
    print(f"Token accuracy: {token_acc}, Sentence accuracy: {sentence_acc}, Total tokens: {total_tokens}, Total sentences: {total_sentences}, Correct tokens: {correct_tokens}, Correct sentences: {correct_sentences}")
    return confusion_matrix

# Main function
def main():
    # Load the CoNLL-2003 dataset
    dataset = load_dataset("conll2003")

    # Access train, validation, and test splits
    test_data = dataset["test"]

    # Preprocess the data
    test_data, test_labels = preprocess(test_data)

    # Parameters
    number_of_tags = 2
    index_to_tag = {0: "O", 1: "B/I"}

    # Load the models
    model_1 = load_model('results/model_1.pkl')
    model_2 = load_model('results/model_2.pkl')
    models = [model_1, model_2]

    # Evaluate the model on the test data
    confusion_matrix = evaluate(models, test_data, test_labels, number_of_tags)
    
    # Calculate Recall, Precision, F1 Score, F0.5 Score, F2 Score
    results = {}

    for i in range(number_of_tags):
        recall = []
        precision = []
        f1 = []
        f0_5 = []
        f2 = []
        tp = confusion_matrix[i][i]
        fp = np.sum(confusion_matrix[i, :]) - tp
        fn = np.sum(confusion_matrix[:, i]) - tp
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1 = 2 * tp / (2 * tp + fn + fp)
        f0_5 = 1.25 * tp / (1.25 * tp + 0.25 * fn + fp)
        f2 = 5 * tp / (5 * tp + 4 * fn + fp)
        results[index_to_tag[i]] = {"recall": recall, "precision": precision, "f1": f1, "f0.5": f0_5, "f2": f2}

    # Print the results for each tag
    for tag, metrics in results.items():
        print(f"Tag: {tag}, Recall: {metrics['recall']}, Precision: {metrics['precision']}, F1 Score: {metrics['f1']}, F0.5 Score: {metrics['f0.5']}, F2 Score: {metrics['f2']}")

    # Heatmap of the confusion matrix
    confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=0)
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(confusion_matrix, annot=True, fmt='.3f', annot_kws={"size": 8}, square=True, cbar=True, cmap = 'viridis', xticklabels=index_to_tag.values(), yticklabels=index_to_tag.values())
    ax.set_xlabel('Actual Tag', labelpad=10, fontsize=10, fontweight='bold')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('Predicted Tag', fontsize=10, fontweight='bold')
    plt.title('Heatmap of Confusion Matrix', fontsize=16, fontweight='bold')
    plt.savefig('results/heatmap.png')

if __name__ == "__main__":
    main()
