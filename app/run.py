import json
import plotly
import pandas as pd
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from flask import Flask, render_template, request, jsonify
from plotly.graph_objs import Bar
from joblib import load
import nltk
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin

app = Flask(__name__)

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    def starting_verb(self, text):
        sentence_list = sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = pos_tag(word_tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return 1
        return 0

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

# Tokenization function
def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    return clean_tokens

# Load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse_table', con=engine)

# Load model
model = load("models/classifier.pkl")

# Index route to display visuals and get user input
@app.route('/')
@app.route('/index')
def index():
    # Data extraction for the first visualization
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Data extraction for the second visualization (top 10 categories)
    category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False).head(10)
    category_names = list(category_counts.index)

    # Visualizations
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Genre"}
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],
            'layout': {
                'title': 'Top 10 Message Categories',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Category"}
            }
        }
    ]

    # Encode Plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# Webpage to handle user query and display model results
@app.route('/go')
def go():
    query = request.args.get('query', '')
    labels = model.predict([query])[0]
    cats = list(df.columns[4:])
    results = dict(zip(cats, labels))

    # sort so positives first, then alphabetically
    results = dict(sorted(results.items(), key=lambda kv: (-kv[1], kv[0])))

    return render_template('go.html', query=query, classification_result=results)


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()
