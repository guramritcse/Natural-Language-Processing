import nltk
from nltk.corpus import brown
from hmm import HMM
from hmm3 import HMM3
import random

# Download Brown corpus and universal tagset
nltk.download('brown')
nltk.download('universal_tagset')

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
    for sentence in test_data:
        words = [word for word, tag in sentence]
        predicted_tags = model.predict(words)
        if predicted_tags == None:
            continue
        actual_tags = [tag for word, tag in sentence]
        all_correct = True
        for predicted_tag, actual_tag in zip(predicted_tags, actual_tags):
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
    test_sentences = len(test_data)
    print(f"Token accuracy: {token_acc}, Sentence accuracy: {sentence_acc}, Total tokens: {total_tokens}, Total sentences: {total_sentences}, Correct tokens: {correct_tokens}, Correct sentences: {correct_sentences}, Test sentences: {test_sentences}")
    return token_acc, sentence_acc

# Main function
def main():
    # Load the dataset
    dataset = load_dataset()

    # Shuffle the dataset
    random.seed(0)
    random.shuffle(dataset)

    # Split the dataset into 5 parts
    data = []
    for i in range(5):
        data.append(dataset[i * len(dataset) // 5:(i + 1) * len(dataset) // 5])

    # Cross-validation
    token_accs = []
    sentence_accs = []
    for i in range(5):
        test_data = data[i]
        train_data = []
        for j in range(5):
            if j != i:
                train_data.extend(data[j])
        # model = HMM()
        model = HMM3()
        model.train(train_data)

        print(f"Fold {i+1}")
        token_acc, sentence_acc = evaluate(model, test_data)
        token_accs.append(token_acc)
        sentence_accs.append(sentence_acc)
    
    # Print the average token and sentence accuracies
    print(f"Average token accuracy: {sum(token_accs) / len(token_accs)}")
    print(f"Average sentence accuracy: {sum(sentence_accs) / len(sentence_accs)}")


if __name__ == "__main__":
    main()
