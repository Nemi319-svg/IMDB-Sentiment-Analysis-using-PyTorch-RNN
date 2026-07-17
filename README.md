# IMDB Sentiment Analysis using PyTorch RNN

An end-to-end Deep Learning pipeline built with PyTorch to classify movie reviews from the IMDB dataset as either positive or negative.

##Dataset
The project utilizes the **IMDB Dataset of 50K Movie Reviews**. Due to GitHub file-size optimization guidelines, please download the raw data directly from Kaggle and place it in the root folder before execution.

##  Pipeline Architecture
1. **Text Preprocessing**: Lowercasing, removing HTML tags/URLs, stripping punctuation, stopword removal via NLTK, and Porter Stemming.
2. **Feature Extraction**: Text text tokenization vectorized using TF-IDF Vectorization restricted to the top 5,000 text features.
3. **Deep Learning Model**: A Custom Recurrent Neural Network (RNN) implemented with PyTorch handling sequential batch data vectors.

##  Model Performance
* **Training Duration**: 10 Epochs
* **Final Model Accuracy**: ~85.15%
