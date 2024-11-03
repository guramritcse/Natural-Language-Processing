import streamlit as st
import pandas as pd
from metric import load_dataset, save_model, load_model
from nltk.tokenize import TreebankWordTokenizer

# Sentence tokenizer
tokenizer = TreebankWordTokenizer()

# Title of the app
st.title("SVM-BASED NEI")

# Load the model
model = load_model('results/model.pkl')

# Input field for user to enter a sentence
sentence_input = st.text_input("Enter a sentence:", "Washington DC is the capital of United States of America​")

# Button to predict POS tags
if st.button("Predict"):
    # Process the sentence
    words = tokenizer.tokenize(sentence_input)
    predicted_tags = model.predict([words])[0].tolist()
    
    # Check if the prediction was successful
    if predicted_tags:
        # Create a DataFrame with the words and their corresponding tags
        df = pd.DataFrame({"Word": words, "NEI_Tag": predicted_tags})
        st.write("Predicted NEI Tags:")
        st.dataframe(df)
    else:
        st.error("Something went wrong. Try again!!!")

