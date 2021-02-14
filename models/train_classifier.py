# Import libraries
import pandas as pd
import numpy as np
import pickle
import sys
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


# Load data from database
def load_from_db(database_name):
    '''
    Load the dataset generated from ETL pipeline

            Returns:
                    X (pd.DataFrame): DataFrame containing the dataset
                    y (pd.DataFrame): Labels of the data
    '''
    engine = create_engine('sqlite:///' + database_name)
    df = pd.read_sql('messages_and_categories', engine)
    X = df.message
    y = df.drop(['message', 'original', 'genre'], axis=1)
    return X, y



def tokenize(text):
    '''
    Tokenize function performs multiple preprocessing steps:
        - Converts text to lower case
        - Replaces any special character
        - Tokenizes text
        - Removes stop words
        - Lemmatize
        - Stem

            Parameters:
                    text (str): a message to be tokenized

            Returns:
                    text (str): Preprocessed and tokenized text
    '''
    # Normalize
    text = text.lower()
    text = text.replace(r"([^a-zA-Z0-9])", ' ')

    # Tokenize
    text = word_tokenize(text)

    # Remove stop words, stem and lemmatize.
    # Those were combined in order not to iterate multiple times
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    text = [stemmer.stem(lemmatizer.lemmatize(w.strip()))
            for w in text if w not in stopwords.words("english")]
    return text


# Display class-level performance metrics for the best model
def display_results(cv, y_test, y_pred):
    '''
    Display the model metrics; Accuracy, F1 score, Precision, Recall
    In case the pipeline is the result of some search algorithm the best_params are displayed.

            Parameters:
                    cv (Pipeline): The pipeline containing the model
                    y_test (numpy.array): Ground truth labels for test data
                    y_pred (numpy.array): Model output predictions
    '''

    for i in range(y_test.shape[1]):
        col = y_test.iloc[:, i]

        print('Label name: ', col.name)
        print('Label values:', np.unique(col))
        print('Accuracy: ', accuracy_score(col, y_pred[:, i]))
        print('F1 score: ', f1_score(col, y_pred[:, i], zero_division=0))
        print('Precision:', precision_score(col, y_pred[:, i], zero_division=0))
        print('recall:', recall_score(col, y_pred[:, i], zero_division=0))
        print('-----------------------')

    # This is in case what was being passed is just a model not as a result of gridserch
    if hasattr(cv, 'best_params_'):
        print("\nBest Model Parameters:", cv.best_params_)


def build_model(search_method=None):
    '''
    Build ml pipeline. In case a search method was specified it will be used to fit the pipeline

            Parameters:
                    search_method (str): The search method to be used if any (default: None)
                        'grid': GridSearchCV
                        'randomized': RandomizedSearchCV
                        None: no search method is to be used

            Returns:
                    cv (Pipeline): The pipeline containing the model
    '''

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-1), n_jobs=-1)),
    ])

    # The parameters were commented out because the execution time was very long on my machine
    # Only kept one to showcase that the implementation works
    parameters = {
        # 'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        # 'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False)
    }

    if search_method == 'grid':
        print('Using GridSearchCV')
        cv = GridSearchCV(pipeline, param_grid=parameters, scoring='f1_micro', verbose=3)
    elif search_method == 'randomized':
        print('Using RandomizedSearchCV')
        cv = RandomizedSearchCV(pipeline, param_distributions=parameters, verbose=2)
    else:
        print('Using the pipeline without a search method')
        # if no grid search is needed, just return the pipeline
        cv = pipeline

    return cv


# Save model to file
def save_model(model, path):
    '''
    Save specified model to path

            Parameters:
                    model (Pipeline): The pipeline containing the model
                    path (str): path to write the pickle file

    '''
    with open(path, 'wb') as file:
        pickle.dump(model, file)


def main():
    # default paths in case no arguments were provided
    database_name = './data/DisasterResponse.db'
    model_file_path = './models/classifier.pkl'


    if len(sys.argv) == 4:
        database_name = sys.argv[1]
        model_file_path = sys.argv[2]

    print('Starting ML pipeline')

    # Load data from database
    print('Loading data from database')
    X, y = load_from_db(database_name)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # build classifier
    print('Fitting model')
    model = build_model(search_method='grid')
    model.fit(X_train, y_train)

    # predict on test data
    print('Generating predictions')
    y_pred = model.predict(X_test)

    # display results
    print('Model scores per class:')
    display_results(model, y_test, y_pred)

    # Get the best model, in case GridSearch/RandomizedSearch was used. Which is the best_estimator_.
    # Otherwise, it is just the model itself
    best_model = model
    if hasattr(model, 'best_estimator_'):
        best_model = model.best_estimator_

    # Save model
    print('Saving model')
    save_model(best_model, model_file_path)

    print('Finished ML pipeline')


if __name__ == '__main__':
    main()
