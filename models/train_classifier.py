import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import classification_report, f1_score, precision_score, recall_score


import pickle
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Define StartingVerbExtractor class
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    def starting_verb(self, text):
        sentence_list = sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(word_tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return 1
        return 0

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

# Load data function
def load_data(database_filepath):
    """
    Loads data from a SQLite database, splits it into features and labels,
    and extracts category names.
    """
    table_name = database_filepath.split('/')[-1].replace('.db', '') + "_table"
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table(table_name, con=engine)
    
    X = df['message']  # feature column
    Y = df.iloc[:, 4:]  # labels start from the 4th column onward
    category_names = Y.columns.tolist()
    
    return X, Y, category_names

# Tokenize function
def tokenize(text):
    # normalize urls, numbers, and punctuation
    text = re.sub(r"http\S+|www\.\S+", " url ", text.lower())
    text = re.sub(r"\d+", " num ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(t) for t in tokens if t not in stopwords.words("english")]

# Build model function
def build_model(fast=True):
    """
    fast=True  -> LinearSVC (very fast, no probabilities)
    fast=False -> LogisticRegression (still fast, gives predict_proba)
    """
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        ngram_range=(1, 2),
        min_df=2,
        max_features=100_000,
        sublinear_tf=True
    )

    if fast:
        base_clf = LinearSVC(class_weight="balanced")
    else:
        base_clf = LogisticRegression(
            solver="liblinear",  # or "saga" if you prefer
            max_iter=2000,
            class_weight="balanced"
        )

    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", OneVsRestClassifier(base_clf, n_jobs=-1))
    ])
    return pipeline

# Evaluate model function
def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    # Per-label report
    for i, col in enumerate(category_names):
        print(f"Category: {col}")
        print(classification_report(Y_test[col], Y_pred[:, i], zero_division=0))
        print("="*60)
    # Global scores
    micro_f1 = f1_score(Y_test.values, Y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(Y_test.values, Y_pred, average="macro", zero_division=0)
    print(f"Micro F1: {micro_f1:.3f} | Macro F1: {macro_f1:.3f}")

# Save model function
def save_model(model, model_filepath):
    """
    Saves a trained model to a specified file using pickle.
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model exported as {model_filepath}")

# Main function
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()
