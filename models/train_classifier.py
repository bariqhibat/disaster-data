import sys
import nltk
import pandas as pd
import string
import re
import joblib

from scipy import stats 
from sqlalchemy import create_engine

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download(['punkt','wordnet', 'averaged_perceptron_tagger'])
nltk.download('stopwords')

def load_data(database_filepath):
    """Load the database

    Args:
        database_filepath ([string]): Database path

    Returns:
        X [object]: X for machine learning model
        y [DataFrame]: y for machine learning model
        category_names [List] : category names from data
    """    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Project', engine)
    engine.dispose()

    X = df['message']
    y = df.drop(['message', 'genre', 'original'], axis=1)
    category_names = y.columns.tolist()

    return X, y, category_names

def tokenize(text):
    """Tokenization of the text

    Args:
        text ([string]): Get the string of the text

    Returns:
        clean_tokens [list]: Export a list of tokens
    """    
    text = re.sub('[^a-zA-Z]',' ',text).lower()
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    clean_tokens = [word for word in clean_tokens if not word in set(stopwords.words('english'))]

    return clean_tokens

def build_model():
    """Build the machine learning models

    Returns:
        CV: Machine Learning Model
    """    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(
            RandomForestClassifier(n_estimators=100)))
    ])
    
    param_grid = {
        'clf__estimator__n_estimators': [10,100],
        'clf__estimator__min_samples_split': [2,4],
        'clf__estimator__max_depth': [2,6],
    }
    
    cv = GridSearchCV(pipeline, param_grid, cv=3, verbose=10)
    
    return cv

def evaluate_model(model, X_test, y_test, category_names):
    """Evaluation of the machine learning model

    Args:
        model: Machine Learning Model
        X_test: X test for the model
        y_test: y test for the model
        category_names: category names of the data
    """    
    y_pred = model.predict(X_test)

    y_pred = pd.DataFrame(y_pred,columns=category_names)

    print(classification_report(y_test, y_pred, target_names=category_names))

def save_model(model, model_filepath):
    """Save the models that has been trained

    Args:
        model: Machine Learning Model
        model_filepath: Filepath for saving
    """    
    joblib.dump(model, model_filepath)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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