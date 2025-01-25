import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load your trained model
model = tf.keras.models.load_model('imdb_rnn_model.h5')

# Tokenizer and max sequence length (you should use the same tokenizer as in training)
tokenizer = Tokenizer(num_words=10000)

# Streamlit UI
st.title("IMDB Movie Review using RNN ")
st.write("Enter a movie review to get its sentiment (positive/negative) prediction.")

# Input text field
review_text = st.text_area("Enter Movie Review")

if st.button("Predict Sentiment"):
    if review_text:
        # Preprocess the input text (tokenize and pad)
        sequences = tokenizer.texts_to_sequences([review_text])
        padded_sequences = pad_sequences(sequences, maxlen=200)

        # Make prediction
        prediction = model.predict(padded_sequences)
        
        # Display the result
        sentiment = "Positive" if prediction > 0.5 else "Negative"
        st.write(f"The sentiment is: {sentiment}")
    else:
        st.write("Please enter a review to get the sentiment prediction.")
