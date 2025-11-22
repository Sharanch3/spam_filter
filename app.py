import pickle
import gzip
import streamlit as st
import pandas as pd
from nlp_preprocessor import nlp_pipeline

# Load pickle files
model = pickle.load(gzip.open('./artifacts/model.pkl.gz', 'rb'))
tfidf = pickle.load(gzip.open('./artifacts/vectorizer.pkl.gz', 'rb'))


st.set_page_config(page_title="Spam Classifier", page_icon="📩", layout='centered')
st.title("📩 Spam Classifier")
st.write("Enter a text message below to check if it's spam or ham (legitimate).")
st.divider()



input_text = st.text_area("Enter the message:", height=150, placeholder="Type your message here...")



if st.button("Classify", type="primary"):
    if input_text.strip():
        with st.spinner("Analyzing message..."):
            # Convert to Series, apply pipeline, then extract the value
            transformed_text = nlp_pipeline(pd.Series([input_text])).iloc[0]
            
            # Transform using TF-IDF vectorizer
            vector = tfidf.transform([transformed_text])
            
            # Make prediction
            result = model.predict(vector)[0]
            
            st.divider()
            if result == 0:
                st.success("✅ **HAM** - This is a legitimate message")
            else:
                st.error("⚠️ **SPAM** - This message appears to be spam")
    else:
        st.warning("⚠️ Please enter a message to classify")