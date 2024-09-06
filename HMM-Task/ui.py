import streamlit as st
import pandas as pd
from metric import load_dataset, save_model, load_model
from hmm3 import HMM3
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import TreebankWordTokenizer

# Sentence tokenizer
tokenizer = TreebankWordTokenizer()

# Title of the app
st.title("HMM-based POS Tagger")

model = load_model('results/model.pkl')

# Input field for user to enter a sentence
sentence_input = st.text_input("Enter a sentence:", "The quick brown fox jumps over the lazy dog")

# Button to predict POS tags
if st.button("Predict"):
    # Process the sentence
    words = sentence_input.lower()
    words = tokenizer.tokenize(words)
    predicted_tags = model.predict(words)
    
    # Check if the prediction was successful
    if predicted_tags:
        # Create a DataFrame with the words and their corresponding tags
        df = pd.DataFrame({"Word": words, "Tag": predicted_tags})
        st.write("Predicted POS Tags:")
        st.dataframe(df)
    else:
        st.error("Some words in the sentence were not found in the training data.")

