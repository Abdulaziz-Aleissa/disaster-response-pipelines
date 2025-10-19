import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(datasetOne, datasetTwo):
    """
    Loads and merges two datasets: one containing messages and the other containing categories.

    Parameters:
    ----------
    datasetOne :
        The file path to the messages dataset in CSV format.
    datasetTwo : 
        The file path to the categories dataset in CSV format.

    Returns:
    -------
        A merged DataFrame containing both messages and their respective categories,
        merged on the 'id' column.

    Notes:
    ------
    - The function assumes that both CSV files contain a common 'id' column to perform the merge.
    """
    
    messages = pd.read_csv(datasetOne)
    categories = pd.read_csv(datasetTwo)
    df = pd.merge(messages,categories,on='id')
    return df 


def clean_data(df):
    # Split 'categories' into separate columns
    categories = df['categories'].str.split(';', expand=True)

    # Use the text before '-' from the first row as column names
    first_row = categories.iloc[0]
    category_colnames = first_row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames

    # Convert each category value to numeric {0,1}
    for col in categories.columns:
        # get the number after the hyphen
        categories[col] = categories[col].str.split('-').str[-1]
        # coerce to numeric
        categories[col] = pd.to_numeric(categories[col], errors='coerce')
        # map 2 -> 1 (common in 'related')
        categories[col] = categories[col].replace(2, 1)
        # clip to [0,1], fill NaN with 0, cast to int
        categories[col] = categories[col].clip(0, 1).fillna(0).astype(int)

    # OPTIONAL: drop columns that are entirely zeros (e.g., 'child_alone' in some datasets)
    zero_cols = [c for c in categories.columns if categories[c].sum() == 0]
    if zero_cols:
        categories = categories.drop(columns=zero_cols)

    # Drop original 'categories' and concat cleaned targets
    df = df.drop(columns=['categories'])
    df = pd.concat([df, categories], axis=1)

    # Remove duplicates
    df = df.drop_duplicates()

    return df





def save_data(df, database_filepath):
    """
    Saves a DataFrame to a specified SQLite database file.

    Parameters:
    ----------
    df : pd.DataFrame
        The combined dataset containing messages and cleaned categories.
    database_filepath : str
        Filepath for the SQLite database (e.g., 'data/MyDatabase.db')
    Notes:
    ------
    - The table name is automatically generated based on the database file name.
    """
    # Create the SQLite engine to connect to the database
    engine = create_engine(f'sqlite:///{database_filepath}')
    
    # Automatically generate a table name based on the database filename
    table_name = f"{database_filepath.split('/')[-1].replace('.db', '')}_table"
    
    # Write the DataFrame to the specified SQLite table
    df.to_sql(table_name, engine, index=False, if_exists='replace') 


def main():
    if len(sys.argv) == 4:

        datasetOne, datasetTwo, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(datasetOne, datasetTwo))
        df = load_data(datasetOne, datasetTwo)

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