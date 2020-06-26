# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/Project.db models/disaster_model.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:3001/

### Files:
```
- app
| - template
| |- index.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app
|- language.csv # Language of the messages

- data
|- disaster_categories.csv  # data to process
|- disaster_messages.csv  # data to process
|- process_data.py
|- Project.db   # database to save clean data to

- models
|- train_classifier.py
|- disaster_model.pkl  # saved model, might not be in this repo since the size is too high

- visualization_trial.ipnyb

- README.md
```

### Required packages:

- flask
- joblib
- pickle
- googletrans
- pandas
- plotly
- numpy
- scikit-learn
- sqlalchemy


### Data Details:

Data used in this project comes from Figure-Eight [disaster response data](https://www.figure-eight.com/dataset/combined-disaster-response-data/). Although it is clean, it still needed some learning and wrangling process before it can be used for machine learning application.

### Improvements and Difficulties:

The data that is provided by Figure Eight isn't extremely broad. In sense that, there are many classes that are only a few occurances that they appear, while some classes always appear in the data. This will introduce bias to the machine learning model and might have to do with Outliers. You can try to do something with these outliers.