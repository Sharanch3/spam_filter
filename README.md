# 📩 SMS Spam Classifier

A machine learning-powered web application that classifies SMS messages as spam or ham (legitimate) using Natural Language Processing and Naive Bayes classification.

## 🌟 Features

- **Real-time Classification**: Instantly classify any SMS message as spam or ham
- **NLP Processing**: Advanced text preprocessing using spaCy
- **High Accuracy**: Trained on a dataset with ~97% accuracy
- **User-Friendly Interface**: Clean and intuitive Streamlit web interface
- **Lightweight Model**: Uses TF-IDF vectorization and Multinomial Naive Bayes

## 🚀 Live Demo

**🌐 Link**: --- https://sms-spamfilter.streamlit.app/

## 📈 Model Training

The model was trained using the following pipeline:

1. **Data Cleaning**
   - Removed duplicates
   - Converted labels to binary (0: ham, 1: spam)
   
2. **Text Preprocessing**
   - Lowercasing
   - Punctuation removal
   - Link removal
   - Tokenization
   - Lemmatization
   - Stop word removal

3. **Feature Engineering**
   - TF-IDF vectorization
   - Minimum document frequency: 5

4. **Model Selection**
   - Tested: Logistic Regression & Multinomial Naive Bayes
   - Selected: Multinomial Naive Bayes (best performance)

For detailed analysis, see `notebook.ipynb`

----


⭐ If you found this project helpful, please consider giving it a star!

Made with ❤️ by Sharanch Mukhia
