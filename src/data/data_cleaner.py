import pandas as pd
import ast


def to_datetime(df, column):
    # filtering of data for panda.dataframe compliances
    df = df[(
    df[column] > '1850-01-01') & (df[column] < '2022-01-01'
    )]
    # handling the nan case
    df[column] = df[column].fillna('1850-01-01')
    # keeping only year
    df[column] = pd.to_datetime(df[column], format="mixed").dt.year
    # converting to int
    df[column] = df[column].apply(lambda x: int(x))

    return df

def get_gender_from_name(name, name_gender_df):
    name_row = name_gender_df[name_gender_df['Name'] == name]
    
    if name_row.empty:
        return None
    else:
        return name_row['Gender'].values[0]
    
def extract_names_from_tuples(languages):
    lang_dict = ast.literal_eval(languages)
    return list(lang_dict.values())

def map_cluster(mapping, elem):
    for generic, variants in mapping.items():
        if elem in variants:
            return generic
    return elem