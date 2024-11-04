import streamlit as st
import pandas as pd
from metric import load_dataset, save_model, load_model
from nltk.tokenize import TreebankWordTokenizer

# Sentence tokenizer
tokenizer = TreebankWordTokenizer()

# Title of the app
st.title("SVM-BASED NEI")

# Load the model
model_1 = load_model('results/model_1.pkl')
model_2 = load_model('results/model_2.pkl')

# Input field for user to enter a sentence
sentence_input = st.text_input("Enter a sentence:", "Seoul is the capital of Republic of Korea")

# Button to predict POS tags
if st.button("Predict"):
    # Process the sentence
    words = tokenizer.tokenize(sentence_input)
    predicted_tags_1 = model_1.predict([words], desc=0, type=1)[0].tolist()
    predicted_tags_2 = model_2.predict([words], desc=0, type=2)[0].tolist()
    predicted_tags = [1 if predicted_tag_1 == 1 or predicted_tag_2 == 1 else 0 for predicted_tag_1, predicted_tag_2 in zip(predicted_tags_1, predicted_tags_2)]
    
    # Check if the prediction was successful
    if predicted_tags:
        # Create a DataFrame with the words and their corresponding tags
        df = pd.DataFrame({"Word": words, "NEI_Tag": predicted_tags})
        st.write("Predicted NEI Tags:")
        st.dataframe(df)
    else:
        st.error("Something went wrong. Try again!!!")

