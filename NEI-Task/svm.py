from sklearn.svm import LinearSVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
import nltk
import string

def is_num(word):
    try:
        _ = float(word)
        return 1
    except:
        return 0
    
def is_first_num(word): 
    return is_num(word[0])

class SVM:

    # Initialize the SVM model
    def __init__(self):
        self.model = Pipeline([
            ("vectorizer", DictVectorizer(sparse=True)),
            ("svm", LinearSVC(random_state=42, max_iter=100000, loss='hinge'))
        ])

    # Extract features for a given sentence
    def get_features(self, words, pos_tags):
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        punctuation = list(string.punctuation)
        stopwords = nltk.corpus.stopwords.words('english')
        features = []
        for i in range(len(words)):
            feature = {
                "word": words[i],
                "lower": words[i].lower(),
                "upper": words[i].upper(),
                "position": (i+1)/len(words),
                "length": len(words[i]),
                "is_punctuation": words[i] in punctuation,
                "is_stopword": words[i].lower() in stopwords,
                "is_num": is_num(words[i]),
                "is_first_num": is_first_num(words[i]),
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
                "pos_tag": pos_tags[i],
                "prev_pos_tag": "START" if i == 0 else pos_tags[i - 1],
                "next_pos_tag": "END" if i == len(words) - 1 else pos_tags[i + 1]
            }
            features.append(feature)
        return features

    # Train the SVM on a given dataset
    def train(self, train_data, train_labels):
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger_eng')
        except LookupError:
            nltk.download('averaged_perceptron_tagger_eng')
        # Get the POS tags for each word
        pos_tags = [tag for sentence in train_data for tag in nltk.pos_tag(sentence)]
        # Extract the words
        words = [word for sentence in train_data for word in sentence]
        # Extract the tags
        tags = [tag for sentence in train_labels for tag in sentence]
        # Extract the features for each word
        features = self.get_features(words, pos_tags)
        # Train the SVM
        self.model.fit(features, tags)

    # Predict the sequence of tags for a given sentence
    def predict(self, sentence):
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger_eng')
        except LookupError:
            nltk.download('averaged_perceptron_tagger_eng')
        pos_tags = [tag for tag in nltk.pos_tag(sentence)]
        return self.model.predict(self.get_features(sentence, pos_tags)).tolist()
        