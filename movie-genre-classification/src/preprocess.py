import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
    """
    Cleans the input text by lowercasing, removing punctuation, and extra whitespace.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    # Remove extra whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_data(df):
    """
    Applies the clean_text function to the description column.
    """
    print("Cleaning text data...")
    df['clean_description'] = df['description'].apply(clean_text)
    return df

def get_tfidf_features(train_df, test_df=None, max_features=10000):
    """
    Vectorizes the cleaned text using TF-IDF.
    Returns the vectorizer, training features, and test features.
    """
    print(f"Extracting TF-IDF features (max_features={max_features})...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features, ngram_range=(1, 2))
    
    # Fit and transform the training data
    X_train = vectorizer.fit_transform(train_df['clean_description'])
    
    # Transform test data if provided
    X_test = None
    if test_df is not None:
        X_test = vectorizer.transform(test_df['clean_description'])
        
    print(f"Feature extraction complete. Vocabulary size: {len(vectorizer.vocabulary_)}")
    return vectorizer, X_train, X_test
