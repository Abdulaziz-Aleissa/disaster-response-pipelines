import json
import plotly
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask, render_template, request, jsonify
from plotly.graph_objs import Bar
from joblib import load
from sqlalchemy import create_engine

app = Flask(__name__)

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
    # Data extraction for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

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
        }
    ]

    # Encode Plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# Webpage to handle user query and display model results
@app.route('/go')
def go():
    # Save user input
    query = request.args.get('query', '')

    # Predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # Render results page with query and classification results
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()
