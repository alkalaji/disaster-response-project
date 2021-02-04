# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_dataset(path, index_col):
    df = None
    try:
        df = pd.read_csv(path, index_col=index_col)
    except:
        print('Error while reading dataset: ' + path)
    return df


def clean_categories(categories: pd.DataFrame):
    # create a dataframe of the 36 individual category columns
    categories = categories.categories.str.split(pat=';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # Removing the numerical part of the column name using str split
    category_colnames = list(row.str.split('-').str[0])

    # rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1 for the newly created columns
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str[1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    return categories


def create_dataframe(messages, categories):
    # merge datasets
    df = pd.merge(messages, categories, how='inner', left_on='id', right_on='id')

    print('Removing duplicates')
    # check number of duplicates
    dups = df.shape[0] - df.drop_duplicates().shape[0]
    if dups > 0:
        # drop duplicates
        df = df.drop_duplicates()
        print('Removed {} duplicates'.format(dups))
    else:
        print('No duplicates found!')
    return df


def write_to_db(df: pd.DataFrame, db_name):
    try:
        engine = create_engine('sqlite:///' + db_name)
        df.to_sql('messages_and_categories', engine, index=False)
    except:
        print('Failed to write to database')


if __name__ == '__main__':
    messages_path = './messages.csv'
    categories_path = './categories.csv'
    database_name = 'DisasterResponse.db'

    if len(sys.argv) == 4:
        messages_path = sys.argv[1]
        categories_path = sys.argv[2]
        database_name = sys.argv[3]

    print('Starting ETL Pipeline')

    # load messages dataset
    print('Loading "{}"'.format(messages_path))
    messages = load_dataset(messages_path, 'id')
    # load categories dataset
    print('Loading "{}"'.format((categories_path)))
    categories = load_dataset(categories_path, 'id')

    # clean categories
    print('Cleaning categories')
    categories = clean_categories(categories)

    # create dataset dataframe
    print('Creating dataset dataframe')
    df = create_dataframe(messages, categories)

    # write to database
    print('Writing to database')
    write_to_db(df, database_name)

    print('Finished ETL pipeline')
