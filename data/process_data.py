#Import libraries
import pandas as pd
import numpy as np
import re
import sys
from sqlalchemy import create_engine

def load_data(msg_path,cat_path):
    """Load the data from their paths

    Args:
        msg_path ([string]): Path to messages.csv
        cat_path ([string]): Path to categories.csv

    Returns:
        df ([DataFrame]): Uncleaned dataframe
    """    
    #Read in file
    messages = pd.read_csv(msg_path)
    categories = pd.read_csv(cat_path)
    df = categories.set_index('id').join(messages.set_index('id'))

    return df

def clean_data(df):
    """Clean the data

    Args:
        df ([DataFrame]): Expected a dataframe argument
    
    Returns:
        df ([DataFrame]): Cleaned dataframe is outputted
    """    
    #Clean data
    categories = df['categories'].str.split(pat=';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.str.split(pat='-').str[0]
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str.split('-').str[-1]
        categories[column] = categories[column].astype(int)

    df.drop(['categories'], 
            axis=1,
            inplace=True)
    df = pd.concat([df, categories], 
                   axis=1)
    df = df[df['related'] != 2]
    df = df.drop_duplicates() 

    #Load to database
    engine = create_engine('sqlite:///Project.db')
    df.to_sql('Project', engine, index=False)
    engine.dispose()
    

def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()