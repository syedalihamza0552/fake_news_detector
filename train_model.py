import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_features_and_labels():
    """
    Load feature matrix and labels from ./dataset/.
    
    Returns:
        scipy.sparse.csr_matrix: Feature matrix
        np.ndarray: Labels
    """
    features = load_npz('./dataset/features_tfidf.npz')
    labels = np.load('./dataset/labels.npy')
    return features, labels

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    Train a model and evaluate its performance.
    
    Args:
        model: Scikit-learn model instance
        X_train, X_test: Training and testing features
        y_train, y_test: Training and testing labels
        model_name (str): Name of the model
    
    Returns:
        dict: Performance metrics
    """
    # Train model
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Compute metrics
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'True'], yticklabels=['Fake', 'True'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'./dataset/{model_name}_confusion_matrix.png')
    plt.close()
    
    return metrics, model

def main():
    # Load features and labels
    print("Loading features and labels from ./dataset/...")
    X, y = load_features_and_labels()
    
    # Split data
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize models
    models = [
        (LogisticRegression(max_iter=1000), 'LogisticRegression'),
        (MultinomialNB(), 'NaiveBayes'),
        (RandomForestClassifier(n_estimators=100, random_state=42), 'RandomForest')
    ]
    
    # Train and evaluate models
    print("Training and evaluating models...")
    results = []
    trained_models = {}
    for model, name in models:
        metrics, trained_model = train_and_evaluate_model(model, X_train, X_test, y_train, y_test, name)
        results.append(metrics)
        trained_models[name] = trained_model
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    print("\nModel Performance Summary:")
    print(results_df)
    results_df.to_csv('./dataset/model_results.csv', index=False)
    
    # Select best model based on F1-score
    best_model_name = results_df.loc[results_df['f1_score'].idxmax()]['model']
    best_model = trained_models[best_model_name]
    
    # Save best model
    print(f"\nSaving best model ({best_model_name}) to ./dataset/best_model.pkl...")
    with open('./dataset/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    print("\nOutput files:")
    print("- ./dataset/model_results.csv")
    print("- ./dataset/best_model.pkl")
    print("- ./dataset/*_confusion_matrix.png (for each model)")

if __name__ == "__main__":
    main()