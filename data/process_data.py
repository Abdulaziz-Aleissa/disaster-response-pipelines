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
    """
    Processes the 'categories' column in the DataFrame by splitting, renaming, 
    extracting integer values, and merging back into the DataFrame.

    This function performs the following transformations:
    1. Splits the 'categories' column in `df` by the semicolon (';') delimiter,
       expanding the results into multiple new columns.
    2. Extracts the prefix from each split entry in the first row to serve as column names.
    3. Renames each new column with the extracted prefix from the original 'categories' entries.
    4. Extracts the last character from each entry in the split columns (assumed to be an integer) 
       and converts it to an integer type.
    5. Drops the original 'categories' column.
    6. Concatenates the new category columns back into the original DataFrame.
    7. Removes duplicate rows.

    Parameters:
    df : The DataFrame containing the 'categories' column to process.

    Returns:
    pandas.DataFrame: The transformed DataFrame with new category columns and duplicates removed.
    """
    # Step 1: Split 'categories' into separate category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # Step 2: Extract new column names for categories
    first_row = categories.iloc[0]
    category_colnames = first_row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames

    # Step 3: Convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].str[-1].astype(int)

    # Step 4: Drop the original 'categories' column from `df`
    df = df.drop('categories', axis=1)

    # Step 5: Concatenate the original `df` with the new `categories` DataFrame
    df = pd.concat([df, categories], axis=1)

    # Step 6: Remove duplicates
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