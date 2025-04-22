import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, save_npz
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_dataset():
    """
    Load the preprocessed dataset from ./dataset/.
    
    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    file_path = './dataset/preprocessed_news.csv'
    df = pd.read_csv(file_path)
    return df

def compute_tfidf_features(texts):
    """
    Convert text to TF-IDF features.
    
    Args:
        texts (Series): Text data to vectorize
    
    Returns:
        scipy.sparse.csr_matrix: TF-IDF feature matrix
        TfidfVectorizer: Fitted vectorizer
    """
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer

def compute_additional_features(df):
    """
    Compute additional features like text length.
    
    Args:
        df (pd.DataFrame): Input DataFrame with 'clean_text'
    
    Returns:
        np.ndarray: Array of additional features
    """
    text_length = df['clean_text'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0).values.reshape(-1, 1)
    return text_length

def main():
    # Load dataset
    print("Loading preprocessed dataset from ./dataset/preprocessed_news.csv...")
    df = load_dataset()
    
    # Ensure clean_text is string type and handle missing values
    df['clean_text'] = df['clean_text'].fillna('').astype(str)
    
    # Compute TF-IDF features
    print("Computing TF-IDF features...")
    tfidf_matrix, vectorizer = compute_tfidf_features(df['clean_text'])
    
    # Compute additional features
    print("Computing additional features...")
    additional_features = compute_additional_features(df)
    
    # Combine features
    print("Combining features...")
    feature_matrix = hstack([tfidf_matrix, additional_features])
    
    # Extract labels
    labels = df['label'].values
    
    # Save features, labels, and vectorizer
    print("Saving features and labels to ./dataset/...")
    save_npz('./dataset/features_tfidf.npz', feature_matrix)
    np.save('./dataset/labels.npy', labels)
    with open('./dataset/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Print summary
    print("\nFeature Engineering Summary:")
    print(f"Total samples: {feature_matrix.shape[0]}")
    print(f"Feature matrix shape: {feature_matrix.shape}")
    print("Output files: ./dataset/features_tfidf.npz, ./dataset/labels.npy, ./dataset/tfidf_vectorizer.pkl")

if __name__ == "__main__":
    main()