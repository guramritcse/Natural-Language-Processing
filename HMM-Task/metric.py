import nltk
from nltk.corpus import brown
from hmm import HMM
from hmm3 import HMM3
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Download Brown corpus and universal tagset
nltk.download('brown')
nltk.download('universal_tagset')

# Define the tagset and the number of tags
tag_to_index = {'ADJ': 0, 'ADP': 1, 'ADV': 2, 'CONJ': 3, 'DET': 4, 'NOUN': 5, 'NUM': 6, 'PRON': 7, 'PRT': 8, 'VERB': 9, '.': 10, 'X': 11}
index_to_tag = {0: 'ADJ', 1: 'ADP', 2: 'ADV', 3: 'CONJ', 4: 'DET', 5: 'NOUN', 6: 'NUM', 7: 'PRON', 8: 'PRT', 9: 'VERB', 10: '.', 11: 'X'}
number_of_tags = 12
number_of_folds = 5

# Load the Brown corpus and convert to the universal tagset
def load_dataset():
    tagged_sentences = brown.tagged_sents(tagset='universal')
    
    # Convert the tagged sentences into a list of tuples (word, tag)
    dataset = []
    for sentence in tagged_sentences:
        sentence_tuples = [(word.lower(), tag) for word, tag in sentence]
        dataset.append(sentence_tuples)
    return dataset

# Evaluate the model on the test data
def evaluate(model, test_data):
    correct_tokens = 0
    total_tokens = 0
    correct_sentences = 0
    total_sentences = 0
    confusion_matrix = np.zeros((number_of_tags, number_of_tags))
    for sentence in test_data:
        words = [word for word, tag in sentence]
        predicted_tags = model.predict(words)
        if predicted_tags == None:
            continue
        actual_tags = [tag for word, tag in sentence]
        all_correct = True
        for predicted_tag, actual_tag in zip(predicted_tags, actual_tags):
            confusion_matrix[tag_to_index[predicted_tag]][tag_to_index[actual_tag]] += 1
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
    confusion_matrix = confusion_matrix / total_tokens
    test_sentences = len(test_data)
    print(f"Token accuracy: {token_acc}, Sentence accuracy: {sentence_acc}, Total tokens: {total_tokens}, Total sentences: {total_sentences}, Correct tokens: {correct_tokens}, Correct sentences: {correct_sentences}, Test sentences: {test_sentences}")
    return token_acc, sentence_acc, confusion_matrix

# Main function
def main():
    # Load the dataset
    dataset = load_dataset()

    # Shuffle the dataset
    random.seed(0)
    random.shuffle(dataset)

    # Split the dataset into number_of_folds parts
    data = []
    for i in range(number_of_folds):
        data.append(dataset[i * len(dataset) // number_of_folds:(i + 1) * len(dataset) // number_of_folds])

    # Cross-validation
    token_accs = []
    sentence_accs = []
    confusion_matrices = []
    for i in range(number_of_folds):
        test_data = data[i]
        train_data = []
        for j in range(number_of_folds):
            if j != i:
                train_data.extend(data[j])
        model = HMM3()
        model.train(train_data)

        print(f"Fold {i+1}")
        token_acc, sentence_acc, confusion_matrix = evaluate(model, test_data)
        token_accs.append(token_acc)
        sentence_accs.append(sentence_acc)
        confusion_matrices.append(confusion_matrix)
    
    # Calculate Recall, Precision, F1 Score, F0.5 Score, F2 Score for each tag
    results = {}
    average_confusion_matrix = np.mean(confusion_matrices, axis=0)

    for i in range(number_of_tags):
        recall = []
        precision = []
        f1 = []
        f0_5 = []
        f2 = []
        for j in range(number_of_folds):
            tp = confusion_matrices[j][i][i]
            fp = np.sum(confusion_matrices[j][i, :]) - tp
            fn = np.sum(confusion_matrices[j][:, i]) - tp
            recall.append(tp / (tp + fn))
            precision.append(tp / (tp + fp))
            f1.append(2 * tp / (2 * tp + fn + fp))
            f0_5.append(1.25 * tp / (1.25 * tp + 0.25 * fn + fp))
            f2.append(5 * tp / (5 * tp + 4 * fn + fp))
        recall = np.mean(recall)
        precision = np.mean(precision)
        f1 = np.mean(f1)
        f0_5 = np.mean(f0_5)
        f2 = np.mean(f2)
        results[index_to_tag[i]] = {"recall": recall, "precision": precision, "f1": f1, "f0.5": f0_5, "f2": f2}

    # Print the average token and sentence accuracies
    print(f"Average token accuracy: {sum(token_accs) / len(token_accs)}")
    print(f"Average sentence accuracy: {sum(sentence_accs) / len(sentence_accs)}")

    # Print the results for each tag
    for tag, metrics in results.items():
        print(f"Tag: {tag}, Recall: {metrics['recall']}, Precision: {metrics['precision']}, F1 Score: {metrics['f1']}, F0.5 Score: {metrics['f0.5']}, F2 Score: {metrics['f2']}")

    # Heatmap of the confusion matrix
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(average_confusion_matrix, annot=True, fmt='.3f', annot_kws={"size": 8}, square=True, cbar=True, cmap = 'viridis', xticklabels=index_to_tag.values(), yticklabels=index_to_tag.values())
    ax.set_xlabel('Actual Tag', labelpad=10, fontsize=10, fontweight='bold')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('Predicted Tag', fontsize=10, fontweight='bold')
    plt.title('Heatmap of Confusion Matrix', fontsize=16, fontweight='bold')
    plt.savefig('results/heatmap.png')

if __name__ == "__main__":
    main()
