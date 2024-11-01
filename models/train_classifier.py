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
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
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
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words("english")]
    return tokens

# Build model function
def build_model():
    """
    Builds a machine learning pipeline for multi-output text classification,
    incorporating StartingVerbExtractor and GridSearchCV for hyperparameter tuning.
    """
    # Defining the initial pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer())
            ])),
            ('starting_verb', StartingVerbExtractor())
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # Defining parameter grid for GridSearch
    param_grid = {
        'clf__estimator__n_estimators': [100, 200, 300],
        'clf__estimator__max_depth': [None, 5, 10],
        'clf__estimator__min_samples_split': [2, 5, 10]
    }

    # Instantiating GridSearchCV with the pipeline and parameter grid
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=2, n_jobs=-1)

    return grid_search

# Evaluate model function
def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the model's performance on the test set by printing the classification report for each category.
    """
    Y_pred = model.predict(X_test)
    Y_pred_df = pd.DataFrame(Y_pred, columns=category_names)
    
    for column in category_names:
        print(f"Category: {column}")
        print(classification_report(Y_test[column], Y_pred_df[column]))
        print("\n" + "="*60 + "\n")

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
