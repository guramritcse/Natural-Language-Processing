from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

class SVM:

    # Initialize the SVM model
    def __init__(self):
        self.model = Pipeline([
            ("vectorizer", DictVectorizer(sparse=True)),
            ("svm", LinearSVC())
        ])

    # Extract features for a given sentence
    def get_features(self, words):
        features = []
        for i in range(len(words)):
            feature = {
                "word": words[i],
                "is_first": i == 0,
                "is_last": i == len(words) - 1,
                "is_capitalized": words[i][0].upper() == words[i][0],
                "is_all_caps": words[i].upper() == words[i],
                "is_all_lower": words[i].lower() == words[i],
                "prefix-1": words[i][0],
                "prefix-2": words[i][:2],
                "prefix-3": words[i][:3],
                "suffix-1": words[i][-1],
                "suffix-2": words[i][-2:],
                "suffix-3": words[i][-3:],
                "prev_word": "" if i == 0 else words[i - 1],
                "next_word": "" if i == len(words) - 1 else words[i + 1],
            }
            features.append(feature)
        return features

    # Train the SVM on a given dataset
    def train(self, train_data, train_labels):
        # Extract the words
        words = [word for sentence in train_data for word in sentence]
        # Extract the tags
        tags = [tag for sentence in train_labels for tag in sentence]
        # Extract the features for each word
        features = self.get_features(words)
        # Train the SVM
        self.model.fit(features, tags)

    # Predict the sequence of tags for a given sentence
    def predict(self, sentence):
        return self.model.predict(self.get_features(sentence)).tolist()
        