import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report




def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("disaster_response",con=engine) 
    X = df['message']
    Y = df.iloc[:,4:]
    return X,Y, Y.columns


def tokenize(text):
    word_tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    new_tokens = []
    for word in word_tokens:
        new_tokens.append(lemmatizer.lemmatize(word).lower().strip())
    return new_tokens


def build_model():
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ]
    )
    parameters = parameters = {
        #'clf__estimator__n_estimators' : [50, 100],
        'clf__estimator__min_samples_split': [2,5],
    }

    cv = GridSearchCV(pipeline,parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    for idx, column in enumerate(category_names):
        report = classification_report(Y_test[column],y_pred[:,idx])
    print(report)


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


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