from src.data.data_loader import load_csv, load_txt, get_wikipedia_infobox_by_id
from src.data.data_cleaner import to_datetime, get_gender_from_name, extract_names_from_tuples, map_cluster
from tqdm import tqdm
import json
import pandas as pd
import numpy as np
DATA_FOLDER_PATH = "data/raw/"

def preprocess_movie_metadata():
    df = load_csv(DATA_FOLDER_PATH + 'movie.metadata.tsv',has_column_names=False, is_tsv=True, column_names=['wikipedia_movie_id', 'freebase_movie_id', 'movie_name', 'movie_release_date',
                            'movie_box_office_revenue', 'movie_runtime', 'movie_languages', 'movie_countries', 'movie_genres'])
    # here, we filter the movies on their date only
    df = to_datetime(df, column='movie_release_date')
    # filtering the years because we have very small data prior to 1910 and 2013
    df = df[(df['movie_release_date'] > 1913) & (df['movie_release_date'] <= 2013)]
    # formatting the languages, countries and genres
    df['movie_languages'] = df['movie_languages'].apply(extract_names_from_tuples)
    df['movie_countries'] = df['movie_countries'].apply(extract_names_from_tuples)
    df['movie_genres'] = df['movie_genres'].apply(extract_names_from_tuples)
    with open('data/raw/clusters.json', 'r') as file:
        data = json.load(file)
    languages_cluster = data['Languages']
    countries_cluster = data['Countries']
    genres_cluster = data['Genres']
    df['movie_languages'] = df['movie_languages'].apply(lambda x: list(dict.fromkeys([map_cluster(languages_cluster, elem) for elem in x]).keys()))
    df['movie_countries'] = df['movie_countries'].apply(lambda x: list(dict.fromkeys([map_cluster(countries_cluster, elem) for elem in x]).keys()))
    df['movie_genres'] = df['movie_genres'].apply(lambda x: list(dict.fromkeys([map_cluster(genres_cluster, elem) for elem in x]).keys()))
    df.to_csv('data/processed/movies_metadata_new.csv', index=False)
    
    return df

def preprocess_char_metadata():

    df = load_csv(DATA_FOLDER_PATH + 'character.metadata.tsv', has_column_names=False, is_tsv=True, column_names=['wikipedia_movie_id', 'freebase_movie_id', 'movie_release_date', 'char_name',
                        'actor_date_of_birth', 'actor_gender', 'actor_height', 'actor_ethnicity', 'actor_name', 'actor_age', 
                        'char_actor_id', 'char_id', 'actor_id'])
    # we convert the columns to datetime
    df = to_datetime(df, column='movie_release_date')

    df = df[(df['movie_release_date'] >= 1913) & (df['movie_release_date'] < 2013)]
    # drop duplicates
    df = df.drop_duplicates(subset=["wikipedia_movie_id", "char_name", "actor_name"]).reset_index(drop=True)
    # drop rows with missing values on both actor_name and actor_gender
    df = df.dropna(subset=["actor_name", "actor_gender"], how="all").reset_index(drop=True)
    # we fill the gender thanks to the actor_name and using an external dataset
    # we load the dataset
    #name_gender = load_csv("data/external/name_gender_dataset.csv")
    name_gender = pd.read_csv("data/external/name_gender_dataset.csv")
    # some names have two gender possible, keep only highest probability
    idx_max_prob = name_gender.groupby('Name')['Probability'].idxmax()
    name_gender = name_gender.loc[idx_max_prob]
    # we take only rows to fill and we apply the function
    gender_nan_actor = df[df['actor_gender'].isna() & df['actor_name'].notna()]
    for index, row in gender_nan_actor.iterrows():
        actor_name = row['actor_name']
        first_name = actor_name.split(" ")[0]
        gender = get_gender_from_name(first_name, name_gender)
        if gender is not None:
            df.at[index, 'actor_gender'] = gender
    # we remove the rows that still have no gender
    df = df.dropna(subset=["actor_gender", "actor_name"]).reset_index(drop=True)
    df.to_csv('data/processed/characters_metadata.csv', index=False)
    return df

def get_movie_ids(movie_df, char_df): 
    """ Take raw dataframes, preprocess them and return the intersection 
    of the wikipedia_movie_id column of both dataframes."""
    
    movie_ids = movie_df['wikipedia_movie_id']
    # in character file, we must drop duplicates
    character_movie_id = char_df.drop_duplicates(subset=['wikipedia_movie_id'])
    # keep only the ids
    character_movie_id = character_movie_id[['wikipedia_movie_id']]
    # merge the two dataframes to get the intersection of the wikipedia_movie_id column
    merged = pd.merge(movie_ids, character_movie_id, how='inner', on='wikipedia_movie_id')
    # we keep only the rows that are in the intersection
    movies_metadata_intersection = movie_df['wikipedia_movie_id'].apply(lambda x : x in merged['wikipedia_movie_id'].values)
    movie_df = movie_df[movies_metadata_intersection]
    # we keep only the rows that are in the intersection
    character_metadata_intersection = char_df['wikipedia_movie_id'].apply(lambda x : x in merged['wikipedia_movie_id'].values)
    char_df = char_df[character_metadata_intersection]
    # storing the intersection ids of wikipedia_movie_id
    ids = movie_df[["wikipedia_movie_id"]]
    
    return ids

def process_imdb_ratings(cmu_df):
    """ cmu_df is the dataframe containing the movies metadata """
    
    # Load the data
    # the title.basics.tsv and title.ratings.tsv files are too big to be stored in the repository
    
    #imdb_title_df = pd.read_csv("src/data/external/title.basics.tsv", sep='\t')
    #imdb_ratings_df = pd.read_csv("src/data/external_data/raw_data/title.ratings.tsv", sep='\t')
    imdb_title_df = load_csv("data/external/title.basics.tsv", is_tsv=True)
    imdb_ratings_df = load_csv("data/external/title.ratings.tsv", is_tsv=True)
    # Merging the two IMDB DataFrames on the 'tconst' column
    imdb_df = pd.merge(imdb_title_df, imdb_ratings_df[['tconst', 'averageRating', 'numVotes']], on='tconst', how='left')
    
    # To be sure to get all names, we consider the two columns primaryTitle and originalTitle
    # We create a match name with the title and the year of the movie, lowercasing the str
    # For IMDB
    imdb_df['primaryTitle_match'] = imdb_df['primaryTitle'].str.lower() + imdb_df['startYear'].str.lower()
    imdb_df['originalTitle_match'] = imdb_df['originalTitle'].str.lower() + imdb_df['startYear'].str.lower()
    imdb_all_titles = pd.concat([imdb_df["primaryTitle_match"], imdb_df["originalTitle_match"]]).drop_duplicates()
    # For CMU
    cmu_titles = cmu_df["movie_name"].str.lower() + cmu_df["movie_release_date"].apply(lambda x: str(x))

    # We get the indexes of the intersection of movies
    intersecting_cmu_titles = cmu_titles.isin(imdb_all_titles)

    # Reshape IMDb data by concatenating both title types into one column along with other columns
    # We drop duplicates on the title
    imdb_ratings_expanded = pd.concat([
        imdb_df[['tconst', 'primaryTitle_match', "startYear", 'averageRating', 'numVotes']].rename(columns={'primaryTitle_match': 'title_match'}),
        imdb_df[['tconst', 'originalTitle_match', "startYear", 'averageRating', 'numVotes']].rename(columns={'originalTitle_match': 'title_match'})
    ]).dropna(subset=['title_match']).drop_duplicates(subset=['title_match'])

    # We keep only the interected df
    cmu_intersection_df = cmu_df.loc[intersecting_cmu_titles]
    # We create the titles match column to merge easily
    cmu_intersection_df['movie_name_match'] = cmu_titles
    
    # We merge
    imdb_in_cmu = cmu_intersection_df.merge(
        imdb_ratings_expanded,
        left_on='movie_name_match',
        right_on='title_match',
        how='left'
    )

    # We drop the rows with no ratings
    imdb_in_cmu = imdb_in_cmu.dropna(subset=["averageRating"])

    # Some movies are duplicated on tconst, we handle them
    # get tconst that are duplicated
    duplicate_tconst_movies = imdb_in_cmu.tconst.value_counts().loc[imdb_in_cmu.tconst.value_counts()!=1]
    # create a transition df with only duplicated movies
    duplicate_df = imdb_in_cmu.loc[imdb_in_cmu.tconst.isin(duplicate_tconst_movies.index.values)]
    # count the number of nans on the rows
    duplicate_df["nan_number"] = duplicate_df.isna().sum(axis=1)
    # chose the row where the number of nan is minimum 
    min_nan = duplicate_df.groupby("tconst")["nan_number"].idxmin()
    # take the opposite to get the movies to remove
    duplicate_to_remove = duplicate_df.index.difference(min_nan)
    # remove the duplicates
    imdb_in_cmu = imdb_in_cmu.drop(duplicate_to_remove, axis=0)
    
    
    # We clean our dataframe to export it
    imdb_in_cmu = imdb_in_cmu[["wikipedia_movie_id", "tconst", "movie_name", "startYear", "averageRating", "numVotes"]].reset_index(drop=True)
    # We rename the columns
    cols_name = ["imdb_movie_id", "movie_release_date"]
    cols_to_rename = ["tconst", "startYear"]
    rename_mapping = dict(zip(cols_to_rename, cols_name))
    imdb_in_cmu.rename(columns=rename_mapping, inplace=True)

    # We are ready to export it
    imdb_in_cmu.to_csv("data/processed/imdb_ratings.csv", index=False)
    
    return imdb_in_cmu

def attribute_gender_to_dir(name_dir, gender_db):
    current_best_prob = 0
    gender = None
    if type(name_dir)!=str:
        name_dir = str(name_dir)
    for string in name_dir.split():
        if string in gender_db['Name'].values :
            prob_str = gender_db[gender_db['Name'] == string]['Probability'].values[0]
            if prob_str > current_best_prob:
                current_best_prob = prob_str
                gender = gender_db[gender_db['Name'] == string]['Gender'].values[0]
    return gender

def get_director_name_and_gender():
    # extract second colomn from data/imdb_ratings.csv
    wiki_movies_id = pd.read_csv('data/imdb_ratings.csv')['wikipedia_movie_id']
    movie_directors = []
    for i in tqdm(wiki_movies_id, desc="Processing movies"):
        infobox_data = get_wikipedia_infobox_by_id(i)
        if (infobox_data is None) or ("Directed by" not in infobox_data):
            continue
    director = infobox_data.get("Directed by")
    movie_directors.append({"wikipedia_movie_id": i, "Director": director})

    movie_directors = pd.DataFrame(movie_directors)
    movie_directors = movie_directors.drop_duplicates()
    gender_db = pd.read_csv("src/data/name_gender_dataset.csv")
    tqdm.pandas()
    movie_directors['Gender'] = movie_directors['Director'].progress_apply(lambda x: attribute_gender_to_dir(x, gender_db))
    movie_directors = movie_directors[movie_directors['Gender'].notna()]
    movie_directors.to_csv('data/movies_director.csv', index=False)

def process_directors():
    movies_director = load_csv('data/processed/movies_director.csv', has_column_names=False, column_names=['wikipedia_movie_id', 'director_name', 'director_gender'])[1:]
    movies_metadata_df = load_csv('data/processed/movies_metadata.csv')
    movies_director['wikipedia_movie_id'] = movies_director['wikipedia_movie_id'].astype(int)
    movies_directors_combined = movies_metadata_df.join(movies_director.set_index('wikipedia_movie_id'), on='wikipedia_movie_id')
    movies_directors_combined = movies_directors_combined[movies_directors_combined['director_gender'].notna()]
    movies_directors_combined.to_csv('data/processed/movies_directors_combined.csv', index=False)
    
def compute_movies_metadata_success():
    # Loading and preprocessing the data of tmdb
    tmdb_ratings = pd.read_csv('src/data/external_data/TMDB_movie_dataset_v11.csv')
    tmdb_ratings = tmdb_ratings[['title', 'vote_average', 'vote_count', 'release_date', 'revenue', 'budget']]
    # We also thought of keeping the popularity column, but we could not retrieve information about the parameters that made a movie "popular" or not, so we decided to drop it.
    # convert 0 values into nan
    tmdb_ratings['revenue'] = tmdb_ratings['revenue'].replace('0', np.nan)
    tmdb_ratings['budget'] = tmdb_ratings['budget'].replace('0', np.nan)
    # We have two different sources (tmdb & imdb) form which we can retrieve the average rating given by people who have watched the movie. In order to decide which dataset we will keep, we will check a specific parameter.
    imdb_ratings = pd.read_csv('data/processed/imdb_ratings.csv')
    # As the analysis of ratings will be truthful if there is a consequent number of ratings in the first place, we decided to pursue our analysis with the ratings of the imdb dataset instead of the tmdb one.
    # If more details needed, imdb_ratings[imdb_ratings['numVotes'] > 50].count() gave us 40084 values, and tmdb_ratings[tmdb_ratings['vote_count'] > 50].count() only 27689 values
    imdb_ratings = imdb_ratings[imdb_ratings['numVotes'] > 50]
    imdb_ratings['numVotes'] = imdb_ratings['numVotes'].astype(int)
    tmdb_ratings = tmdb_ratings.drop(['vote_count', 'vote_average'], axis=1)
    # to avoid matching errors, we will merge datasets based on the name of the movie AND the release year, so weneed to drop any NaN value
    tmdb_ratings = tmdb_ratings.dropna(subset=['release_date'])
    # only keep the release year
    tmdb_ratings['release_date'] = tmdb_ratings['release_date'].apply(lambda x: x if type(x)==float else x.split('-')[0])
    tmdb_ratings['release_date'] = tmdb_ratings['release_date'].astype(int)
    # Many revenues and budgets values of the dataframe are 0s, in order to prevent those values from falsing our analysis results we will convert them into NaN values.
    tmdb_ratings['revenue'] = tmdb_ratings['revenue'].apply(lambda x: np.NaN if x==0 else x)
    tmdb_ratings['budget'] = tmdb_ratings['budget'].apply(lambda x: np.NaN if x==0 else x)
    # Before merging the two datasets, we remove any useless column of imdb ratings and rename the remaining ones.
    imdb_ratings.drop(columns=['imdb_movie_id'], inplace=True)
    imdb_ratings.columns = ['wikipedia_movie_id', 'movie_name', 'release_date', 'average_rating', 'num_votes']
    tmdb_ratings.rename(columns={'title': 'movie_name'}, inplace=True)
    movies_metadata_success = pd.merge(tmdb_ratings, imdb_ratings, on=['movie_name', 'release_date'], how='inner')
    # renaiming the columns
    movies_metadata_success = movies_metadata_success[['wikipedia_movie_id','movie_name', 'release_date', 'revenue', 'budget', 'average_rating', 'num_votes']]
    # then, we pursue the merging with movies metadata df
    movies_metadata = pd.read_csv('data/processed/movies_metadata.csv')
    movies_metadata_success = pd.merge(movies_metadata_success, movies_metadata, how='inner', on=['wikipedia_movie_id'])
    movies_metadata_success = movies_metadata_success[['wikipedia_movie_id', 'movie_name_x', 'release_date_x', 'revenue', 'budget', 'average_rating', 'num_votes', 'genres', 'countries']]
    movies_metadata_success.rename(columns={'movie_name_x': 'movie_name', 'release_date_x': 'release_date'}, inplace=True)
    # the next step is to merge the resulted dataframe with the movies directors df
    movies_directors = pd.read_csv('data/processed/movies_director.csv')
    movies_metadata_success = pd.merge(movies_metadata_success, movies_directors, how='inner', on=['wikipedia_movie_id'])
    movies_metadata_success.drop(columns=['Director'], inplace=True)
    movies_metadata_success.rename(columns={'Gender':'director_gender'}, inplace=True)
    # we can finally export our csv
    movies_metadata_success.to_csv('data/processed/movies_metadata_success.csv', index=False)