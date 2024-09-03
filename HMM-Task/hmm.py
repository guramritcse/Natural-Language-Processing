import math

class HMM:

    # Initialize the HMM
    def __init__(self):
        self.transition_probs = {}
        self.observation_probs = {}
        self.tagset = set()
        self.wordset = set()
        self.start_tag = "^START^_^TAG^"
        self.end_tag = "^END^_^TAG^"


    # Train the HMM on a given dataset
    def train(self, train_data):
        # Initialize the transition and observation probabilities
        self.transition_probs = {}
        self.observation_probs = {}
        self.tagset = set()
        self.wordset = set()

        # Count the occurrences of each tag and word
        tag_counts = {}
        word_counts = {}

        # Add the start and end tags to the tagset
        self.tagset.add(self.start_tag)
        self.tagset.add(self.end_tag)
        for sentence in train_data:
            # Update the tag counts and word counts for the start tag
            if self.start_tag not in tag_counts:
                tag_counts[self.start_tag] = 0
            tag_counts[self.start_tag] += 1
            if self.start_tag not in word_counts:
                word_counts[self.start_tag] = 0
            word_counts[self.start_tag] += 1

            # Update the observation probabilities for the start tag
            if self.start_tag not in self.observation_probs:
                self.observation_probs[self.start_tag] = {}
            if self.start_tag not in self.observation_probs[self.start_tag]:
                self.observation_probs[self.start_tag][self.start_tag] = 0
            self.observation_probs[self.start_tag][self.start_tag] += 1

            prev_tag = self.start_tag
            for word, tag in sentence:
                # Update the tag counts
                if tag not in tag_counts:
                    tag_counts[tag] = 0
                tag_counts[tag] += 1

                # Update the word counts
                if word not in word_counts:
                    word_counts[word] = 0
                word_counts[word] += 1

                # Update the transition probabilities
                if prev_tag not in self.transition_probs:
                    self.transition_probs[prev_tag] = {}
                if tag not in self.transition_probs[prev_tag]:
                    self.transition_probs[prev_tag][tag] = 0
                self.transition_probs[prev_tag][tag] += 1

                # Update the observation probabilities
                if tag not in self.observation_probs:
                    self.observation_probs[tag] = {}
                if word not in self.observation_probs[tag]:
                    self.observation_probs[tag][word] = 0
                self.observation_probs[tag][word] += 1

                # Update the tagset and wordset
                self.tagset.add(tag)
                self.wordset.add(word)

                prev_tag = tag
            
            # Update the tag counts and word counts for the end tag
            if self.end_tag not in tag_counts:
                tag_counts[self.end_tag] = 0
            tag_counts[self.end_tag] += 1
            if self.end_tag not in word_counts:
                word_counts[self.end_tag] = 0
            word_counts[self.end_tag] += 1

            # Update the transition probabilities for the end tag
            if prev_tag not in self.transition_probs:
                self.transition_probs[prev_tag] = {}
            if self.end_tag not in self.transition_probs[prev_tag]:
                self.transition_probs[prev_tag][self.end_tag] = 0
            self.transition_probs[prev_tag][self.end_tag] += 1

            # Update the observation probabilities for the end tag
            if self.end_tag not in self.observation_probs:
                self.observation_probs[self.end_tag] = {}
            if self.end_tag not in self.observation_probs[self.end_tag]:
                self.observation_probs[self.end_tag][self.end_tag] = 0
            self.observation_probs[self.end_tag][self.end_tag] += 1

            # Set the transition probabilities from the end tag to zero
            self.transition_probs[self.end_tag] = {}

        # Normalize the transition and observation probabilities and take the log
        for prev_tag in self.transition_probs:
            total_transitions = sum(self.transition_probs[prev_tag].values())
            for tag in self.transition_probs[prev_tag]:
                self.transition_probs[prev_tag][tag] /= total_transitions
                self.transition_probs[prev_tag][tag] = math.log(self.transition_probs[prev_tag][tag])

        for tag in self.observation_probs:
            total_observations = sum(self.observation_probs[tag].values())
            for word in self.observation_probs[tag]:
                self.observation_probs[tag][word] /= total_observations
                self.observation_probs[tag][word] = math.log(self.observation_probs[tag][word])


    # Predict the sequence of tags for a given sentence
    def predict(self, sentence):
        # Initialize the Viterbi algorithm
        viterbi = [{tag: {"prob": -math.inf, "prev": None} for tag in self.tagset} for _ in range(len(sentence)+1)]
        for tag in self.tagset:
            viterbi[0][tag]["prob"] = self.transition_probs[self.start_tag].get(tag, -math.inf) 
            viterbi[0][tag]["prev"] = self.start_tag

        # Run the Viterbi algorithm
        for t in range(1, len(sentence)+1):
            for tag in self.tagset:
                max_prob = -math.inf
                max_prev_tag = None
                for prev_tag in self.tagset:
                    if sentence[t-1] not in self.wordset:
                        prob = viterbi[t - 1][prev_tag]["prob"] + self.transition_probs[prev_tag].get(tag, -math.inf)
                    else:
                        prob = viterbi[t - 1][prev_tag]["prob"] + self.observation_probs[prev_tag].get(sentence[t-1], -math.inf) + self.transition_probs[prev_tag].get(tag, -math.inf)
                    if prob > max_prob:
                        max_prob = prob
                        max_prev_tag = prev_tag
                viterbi[t][tag]["prob"] = max_prob
                viterbi[t][tag]["prev"] = max_prev_tag

        # Backtrack to find the best sequence of tags
        predicted_tags = []
        current_tag = self.end_tag
        for t in range(len(sentence), 0, -1):
            current_tag = viterbi[t][current_tag]["prev"]
            predicted_tags.append(current_tag)
        predicted_tags.reverse()

        return predicted_tags