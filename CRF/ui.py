import streamlit as st
import pandas as pd
from metric import load_dataset, save_model, load_model
from crf import CRF
from nltk.tokenize import TreebankWordTokenizer

# Sentence tokenizer
tokenizer = TreebankWordTokenizer()

# Title of the app
st.title("CRF-based POS Tagger")

# Load the model
crf_model = load_model('results/model.pkl')
hmm_model = load_model('results/hmm_model.pkl')


# Input field for user to enter a sentence
sentence_input = st.text_input("Enter a sentence:", "The quick brown fox jumps over the lazy dog")

# Button to predict POS tags
if st.button("Predict"):
    # Process the sentence
    words1 = sentence_input
    words1 = tokenizer.tokenize(words1)
    words = []
    for w in words1:
        words.append([w,'X'])

    crf_predicted_tags = crf_model.predict(words).tolist()
    hmm_predicted_tags = hmm_model.predict([wd.lower() for wd in words1])

    
    # Check if the prediction was successful
    if crf_predicted_tags and hmm_predicted_tags:
        # Create a DataFrame with the words and their corresponding tags
        df = pd.DataFrame({"Word": [wd[0] for wd in words], "CRF Tag": crf_predicted_tags, "HMM Tag": hmm_predicted_tags })
        st.write("Predicted POS Tags:")
        st.dataframe(df)
    else:
        st.error("Invalid input.")

