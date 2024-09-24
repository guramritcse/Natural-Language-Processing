import math
import sklearn_crfsuite

def is_num(word):
    try:
        s = float(word)
        return 1
    except:
        return 0
    
def is_first_num(word): return is_num(word[0])
    
def is_Cap(word): return word[0].isupper()

def orthographic_feature(word): return {suff: word.endswith(suff) for suff in ['ing', 'ogy', 'ed', 's', 'ly', 'ion', 'tion', 'ity', 'ies']}

class CRF:

    # Initialize the HMM
    def __init__(self):
        self.transition_probs = {}
        self.observation_probs = {}
        self.tagset = set()
        self.wordset = set()
        self.start_tag = "^START^_^TAG^"
        self.end_tag = "^END^_^TAG^"
        self.punc_tag = "."
        self.small_prob = -math.inf
        self.crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=200,
            all_possible_transitions=True   
        )


    # features for a given sentence
    def word_features(self, sent, i):

        is_cap = is_Cap(sent[i][0])

        word = sent[i][0].lower()
        
        # first word and tag
        if i==0:
            prevword = '<START>'
            prevpos = self.start_tag
        else:
            prevword = sent[i-1][0].lower()
            prevpos = sent[i-1][1]
            
        # first word and tag
        if i==0 or i==1:
            prev2word = '<START>'
            prev2pos = self.start_tag
        else:
            prev2word = sent[i-2][0].lower()
            prev2pos = sent[i-2][1]
        
        # last word and tag
        if i == len(sent)-1:
            nextword = '<END>'
            nextpos = self.end_tag
        else:
            nextword = sent[i+1][0].lower()
            nextpos = sent[i+1][1]

        # suffixes and prefixes of current tag
        pref_1, pref_2, pref_3, pref_4 = word[:1], word[:2], word[:3], word[:4]
        suff_1, suff_2, suff_3, suff_4 = word[-1:], word[-2:], word[-3:], word[-4:]
        
        features = {'word':word,            
                'prevword': prevword,  
                'prev2word': prev2word,      
                'nextword': nextword, 
                'is_num': is_num(word), 
                'is_start_num': is_first_num(word),
                'is_cap': is_cap, 
                'suff_1': suff_1,  
                'suff_2': suff_2,  
                'suff_3': suff_3,  
                'suff_4': suff_4, 
                'pref_1': pref_1,  
                'pref_2': pref_2,  
                'pref_3': pref_3, 
                'pref_4': pref_4
            }
        features.update(orthographic_feature(word))
    
        return features
    

    def sent2features(self, sent):
        return [self.word_features(sent, i) for i in range(len(sent))]

    def sent2labels(self, sent):
        return [postag for _, postag in sent]

    def sent2words(self, sent):
        return [word for word, _ in sent]  
    


    # Train the HMM on a given dataset
    def train(self, train_data):

        X_train = [self.sent2features(s) for s in train_data]
        y_train = [self.sent2labels(s) for s in train_data]
        
        self.crf.fit(X_train, y_train)
                    

    # Predict the sequence of tags for a given sentence
    def predict(self, sentence):
        X_test = self.sent2features(sentence)
        Y_pred = self.crf.predict([X_test])

        return Y_pred[0]