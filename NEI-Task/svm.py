from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
import nltk
import string
import gensim.downloader as api
import numpy as np
from tqdm import tqdm

# Function to check if a word is a number
def is_num(word):
    try:
        _ = float(word)
        return 1
    except:
        return 0

# Class for the SVM model
class SVM:

    # Initialize the SVM model
    def __init__(self):
        self.model = Pipeline([
            ("vectorizer", DictVectorizer(sparse=True)),
            ("svm", LinearSVC(random_state=42, max_iter=100000))
        ])

    # Extract features for a given sentence
    def get_features(self, positions, words, pos_tags, type):
        try:
            nltk.data.find('corpora/words')
        except LookupError:
            nltk.download('words')
        punctuation = set(string.punctuation)
        nltk_words = set(word.lower() for word in nltk.corpus.words.words())
        features = []
        # Extract features for each word
        for i in range(len(words)):
            feature = {
                "word": words[i],
                "prev_word": "" if positions[i][0]==0 else words[i - 1],
                "next_word": "" if positions[i][0]==positions[i][1]-1 else words[i + 1],
                "position": (positions[i][0] / (positions[i][1]-1)) if positions[i][1]>1 else 1,
                "length": len(words[i]),
                "is_punctuation": words[i] in punctuation,
                "is_nltk_word": words[i] in nltk_words,
                "is_num": is_num(words[i]),
                "is_first_num": is_num(words[i][0]),
                "is_first": positions[i][0]==0,
                "is_last": positions[i][0]==positions[i][1]-1,
                "prefix-1": words[i][0],
                "prefix-2": words[i][:2],
                "prefix-3": words[i][:3],
                "prefix-4": words[i][:4],
                "prefix-5": words[i][:5],
                "suffix-1": words[i][-1],
                "suffix-2": words[i][-2:],
                "suffix-3": words[i][-3:],
                "suffix-4": words[i][-4:],
                "suffix-5": words[i][-5:],
                "pos_tag": pos_tags[i][1],
                "prev_pos_tag": "START" if positions[i][0]==0 else pos_tags[i - 1][1],
                "next_pos_tag": "END" if positions[i][0]==positions[i][1]-1 else pos_tags[i + 1][1]
            }
            if type == 2:
                feature.update({"lower": words[i].lower(),
                "upper": words[i].upper(),
                "is_capitalized": words[i][0].upper() == words[i][0],
                "is_all_caps": words[i].upper() == words[i],
                "is_all_lower": words[i].lower() == words[i]})
            features.append(feature)
        return features

    # Train the SVM on a given dataset
    def train(self, train_data, train_labels, type=1):
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger_eng')
        except LookupError:
            nltk.download('averaged_perceptron_tagger_eng')
        if type == 1:
            # Covert train data to lower case
            train_data = [[word.lower() for word in sentence] for sentence in train_data]
        # Get the POS tags for each word using the NLTK POS tagger
        pos_tags = [tag for sentence in train_data for tag in nltk.pos_tag(sentence)]
        # Positions of the words in the sentences
        positions = [(i, len(sentence)) for sentence in train_data for i in range(len(sentence))]
        # Extract the words
        words = [word for sentence in train_data for word in sentence]
        # Extract the tags
        tags = [tag for sentence in train_labels for tag in sentence]
        # Extract the features for each word
        print("Extracting train features...")
        features = self.get_features(positions, words, pos_tags, type)
        # Train the SVM
        print("Training SVM...")
        self.model.fit(features, tags)
        print("Training complete...")
    
    # Predict the sequence of tags for a given sentence
    def predict(self, test_data, desc=0, type=1):
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger_eng')
        except LookupError:
            nltk.download('averaged_perceptron_tagger_eng')
        if type == 1:
            # Covert test data to lower case
            test_data = [[word.lower() for word in sentence] for sentence in test_data]
        # Get the POS tags for each word using the NLTK POS tagger
        pos_tags = [tag for sentence in test_data for tag in nltk.pos_tag(sentence)]
        # Positions of the words in the sentences
        positions = [(i, len(sentence)) for sentence in test_data for i in range(len(sentence))]
        # Extract the words
        words = [word for sentence in test_data for word in sentence]
        # Extract the features for each word
        if desc:
            print("Extracting test features...")
        features = self.get_features(positions, words, pos_tags, type)
        # Predict the tags
        if desc:
            print("Predicting tags...")
        predictions = self.model.predict(features)
        # Split the predictions into sentences
        predicted_tags = []
        start = 0
        for sentence in test_data:
            end = start + len(sentence)
            predicted_tags.append(predictions[start:end])
            start = end
        return predicted_tags
        