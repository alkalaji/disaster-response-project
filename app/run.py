import json
import plotly
import pandas as pd
import plotly.express as pex
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap
# from sklearn.externals import joblib # outdated
# import joblib
import pickle
from sqlalchemy import create_engine

app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages_and_categories', engine)

# load model
# I was having an issue with joblib causing exceptions, so I directly used pickle insted
# I know this is not a best practice bcause you don't want to halt your
# page loading to do some heavy background task. But his is just a work around for now
# model = joblib.load("../models/classifier.pkl")
model = pickle.load(open("../models/classifier.pkl", "rb"))


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Get top 5 labels and their counts
    labels_df = df.iloc[:, 3:]
    top_5_labels = labels_df.sum().sort_values(ascending=False).head(5)
    top_label_names = list(top_5_labels.index)
    top_label_counts = list(top_5_labels.values)

    # Generate correlation matrix for the top 5 features in the dataset
    top_5_corr = df.loc[:, top_label_names].corr()
    fig = pex.bar(top_5_corr)


    # create visuals
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
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=top_label_counts,
                    y=top_label_names,
                    orientation='h'
                )
            ],

            'layout': {
                'title': 'Top 5 message labels in the dataset',
                'yaxis': {
                    'title': "Label"
                },
                'xaxis': {
                    'title': "Count"
                }
            }
        },
        {
            'data': [
                Heatmap(
                    x=top_label_names,
                    y=top_label_names,
                    z=top_5_corr
                )
            ],

            'layout': {
                'title': 'Correlation matrix for the top 5 labels',
                'yaxis': {
                    'title': "Label"
                },
                'xaxis': {
                    'title': "Label"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()