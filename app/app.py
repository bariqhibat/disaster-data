#Import libraries
import json
import joblib
import numpy as np
import plotly
import pandas as pd
import re
import nltk

import plotly.express as px
from plotly.graph_objects import Bar

from flask import Flask
from flask import render_template, request, jsonify
from sqlalchemy import create_engine

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

from googletrans import Translator

#Declare translator
translator = Translator()

#Declare app
app = Flask(__name__)

#Download nltk files
nltk.download(['punkt','wordnet', 'averaged_perceptron_tagger'])
nltk.download('stopwords')

#Tokenize function
def tokenize(text):
    """Tokenize words

    Returns:
        [tokens]: return a list of tokens
    """    
    text = re.sub('[^a-zA-Z]','',text).lower()
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    clean_tokens = [word for word in clean_tokens if not word in set(stopwords.words('english'))]

    return clean_tokens

#Read data from database
engine = create_engine('sqlite:///../data/Project.db')
df = pd.read_sql_table('Project', engine)
engine.dispose()

#Read language data
df_lang = pd.read_csv('language.csv')

#Load pickle models
model = joblib.load("../models/disaster_model.pkl")

def create_figures():
    """Create figures for plotly

    Returns:
        List of Figures: Return a list of figures which corresponds to plotly
    """    
    graph_one = []
    graph_one.append(Bar(x = list(df.groupby('genre').count()['message'].sort_values().index), 
                         y=df.groupby('genre').count()['message'].sort_values()))
    layout_one = dict(title = 'Genres of Words', 
                      xaxis = dict(title='Genre'), 
                      yaxis = dict(title='Number of Messages'))
    
    graph_two = []
    graph_two.append(Bar(x =['Not English','English'], 
                         y=[df.original.isna().sum(),
                            len(df)-df.original.isna().sum()]))
    layout_two = dict(title = 'Translated Messages', 
                      xaxis = dict(title='Translation'), 
                      yaxis = dict(title='Number of Messages'))
    
    graph_three = []
    graph_three.append(Bar(x =df_lang.lang[:10], 
                         y=df_lang.counts[:10]))
    layout_three = dict(title = 'Messages\' Language', 
                      xaxis = dict(title='Language'), 
                      yaxis = dict(title='Number of Messages'))
    
    
    figures = []
    figures.append(dict(data=graph_one, 
                        layout=layout_one))
    figures.append(dict(data=graph_two, 
                        layout=layout_two))
    figures.append(dict(data=graph_three, 
                        layout=layout_three))
    # figures.append(dict(data=graph_four, layout=layout_four))
    
    
    return figures

@app.route('/')
@app.route('/index')
def index():
    """Route to / and /index

    Returns:
        render_template: render_template to index.html with a bunch of datas transferred
    """    
    figures = create_figures()
    
    ids = ['figure-{}'.format(i) for i,_ in enumerate(figures)]
    
    figuresJSON = json.dumps(figures,cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('index.html', ids=ids, figuresJSON= figuresJSON)


@app.route('/go')
def go():
    """Route to /go when predicting a message

    Returns:
        render_template: render_template to go.html with a bunch of datas transferred
    """    
    
    #Get data from query
    query = request.args.get('query', '')

    #If the queried data is in english
    if translator.translate(query).src == 'en':
        #Predict
        classification_labels = model.predict([query])[0]
        classification_results = dict(zip(df.columns[4:], classification_labels))
        return render_template(
            'go.html',
            query=query,
            classification_result=classification_results
        )
    else: #If not in english
        #Translate queried message
        translated_q = translator.translate(query).text
        #Predict
        classification_labels = model.predict([translated_q])[0]
        classification_results = dict(zip(df.columns[4:], classification_labels)) 
        return render_template(
            'go.html',
            query=query,
            translated_q= translated_q,
            classification_result=classification_results
        )

def main():
    app.run(host='localhost', port=3001, debug=True)

if __name__ == '__main__':
    main()