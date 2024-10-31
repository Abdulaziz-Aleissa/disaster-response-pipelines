import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle

def load_data(database_filepath):
    """
    Loads data from a SQLite database, splits it into features and labels,
    and extracts category names.

    Parameters:
    ----------
    database_filepath : 
        Filepath for the SQLite database (e.g., 'data/DisasterResponse2.db').

    Returns:
    -------
    tuple
        X (pd.Series): Features, i.e., the message data.
        Y (pd.DataFrame): Labels for each category.
        category_names (list): Names of the categories in Y.
    """
    # Derive the table name from the database filename
    table_name = database_filepath.split('/')[-1].replace('.db', '') + "_table"
    engine = create_engine(f'sqlite:///{database_filepath}')
    
    # Load the data from the database
    df = pd.read_sql_table(table_name, con=engine)
    
    # Split into features and labels
    X = df['message']  # feature column
    Y = df.iloc[:, 4:]  #labels start from the 4th column onward
    
    # Extract category names
    category_names = Y.columns.tolist()
    
    return X, Y, category_names




def tokenize(text):
    # Remove punctuation and lowercase text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stop words and lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words("english")]
    
    return tokens


def build_model():
    """
    Builds a machine learning pipeline for multi-output text classification.

    Returns:
    -------
    Pipeline
        A scikit-learn pipeline object configured with a vectorizer, TF-IDF transformer,
        and a multi-output classifier using a RandomForestClassifier.
    
    Notes:
    ------
    - The pipeline uses CountVectorizer and TfidfTransformer for text preprocessing.
    - MultiOutputClassifier wraps a RandomForestClassifier for handling multiple labels.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=50, max_depth=10)))  # Reduced depth
    ])
    return pipeline


from sklearn.metrics import classification_report
import pandas as pd

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the model's performance on the test set by printing the classification report for each category.

    Parameters:
    ----------
    model : sklearn.Pipeline
        The trained model pipeline.
    X_test : pd.Series
        The test features (messages) to evaluate the model on.
    Y_test : pd.DataFrame
        The true labels for the test set.
    category_names : list of str
        List of category names corresponding to the labels.
    
    Returns:
    -------
    None
        Prints the classification report for each category.
    
    Notes:
    ------
    - Converts predictions to a DataFrame for easy comparison with `Y_test`.
    - Uses `classification_report` to display precision, recall, and F1-score for each category.
    """
    # Predict on the test set
    Y_pred = model.predict(X_test)
    
    # Convert predictions to DataFrame for easier handling
    Y_pred_df = pd.DataFrame(Y_pred, columns=category_names)
    
    # Iterate through each category and report performance metrics
    for column in category_names:
        print(f"Category: {column}")
        print(classification_report(Y_test[column], Y_pred_df[column]))
        print("\n" + "="*60 + "\n")


def save_model(model, model_filepath):
    """
    Saves a trained model to a specified file using pickle.

    Parameters:
    ----------
    model : estimator
        The trained model to be saved.
    filename : str
        The file path where the model will be saved (e.g., 'classifier_model.pkl').

    Returns:
    -------
    None
        Saves the model as a pickle file and prints a confirmation message.
    
    Notes:
    ------
    - Uses 'wb' mode to write the file in binary format.
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model exported as {model_filepath}")


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