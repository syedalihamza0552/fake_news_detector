import pandas as pd
import nltk
import re
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

def load_and_combine_datasets(true_path, fake_path):
    """
    Load True.csv and Fake.csv, add label column, and combine into a single DataFrame.
    
    Args:
        true_path (str): Path to True.csv
        fake_path (str): Path to Fake.csv
    
    Returns:
        pd.DataFrame: Combined dataset with label column
    """
    # Load datasets
    true_df = pd.read_csv(true_path)
    fake_df = pd.read_csv(fake_path)
    
    # Add label column
    true_df['label'] = 1  # True news
    fake_df['label'] = 0  # Fake news
    
    # Combine datasets
    combined_df = pd.concat([true_df, fake_df], ignore_index=True)
    
    return combined_df

def preprocess_text(text):
    """
    Preprocess text by cleaning, tokenizing, removing stopwords, and lemmatizing.
    
    Args:
        text (str): Input text to preprocess
    
    Returns:
        str: Preprocessed text
    """
    # Handle non-string input
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs, HTML tags, and special characters
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into a string
    return ' '.join(tokens)

def preprocess_dataset(df, sample_fraction=None):
    """
    Preprocess the dataset and optionally sample a fraction of it.
    
    Args:
        df (pd.DataFrame): Input DataFrame with 'title', 'text', 'subject', 'date', 'label'
        sample_fraction (float, optional): Fraction of dataset to sample (e.g., 0.1 for 10%)
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """
    # Drop rows with missing text
    df = df.dropna(subset=['text'])
    
    # Remove duplicates based on text
    df = df.drop_duplicates(subset=['text'])
    
    # Sample a fraction of the dataset if specified
    if sample_fraction is not None and 0 < sample_fraction < 1:
        df = df.sample(frac=sample_fraction, random_state=42)
    
    # Preprocess title and text columns
    df['clean_title'] = df['title'].apply(preprocess_text)
    df['clean_text'] = df['text'].apply(preprocess_text)
    
    return df

def main():
    true_news_path, fake_news_path = "./dataset/True.csv", "./dataset/Fake.csv"
    fraction = None # Set to None to use the entire dataset or 0.1 for 10% of the dataset
    combined_data_path = "./dataset/preprocessed_news.csv"
    # check if the dataset exists then delete it
    try:
        os.remove(combined_data_path)
    except OSError:
        pass
    
    
    # Load and combine datasets
    print("Loading and combining datasets...")
    combined_df = load_and_combine_datasets(true_news_path, fake_news_path)
    
    # Preprocess dataset
    print("Preprocessing dataset...")
    preprocessed_df = preprocess_dataset(combined_df, fraction)
    
    # Save preprocessed dataset
    print(f"Saving preprocessed dataset to {combined_data_path}...")
    preprocessed_df.to_csv(combined_data_path, index=False)
    
    # Print summary
    print("\nDataset Summary:")
    print(f"Total samples: {len(preprocessed_df)}")
    print(f"True news: {len(preprocessed_df[preprocessed_df['label'] == 1])}")
    print(f"Fake news: {len(preprocessed_df[preprocessed_df['label'] == 0])}")
    print(f"Columns: {list(preprocessed_df.columns)}")

if __name__ == "__main__":
    main()