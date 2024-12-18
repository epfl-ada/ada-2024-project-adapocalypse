from src.data.data_loader import load_csv, load_txt, get_wikipedia_infobox_by_id
from src.data.data_cleaner import to_datetime, get_gender_from_name, extract_names_from_tuples, map_cluster
from tqdm import tqdm
tqdm.pandas()
import json
import pandas as pd
import numpy as np
import ast
import os
#from transformers import pipeline
from collections import defaultdict


RAW_DATA_FOLDER_PATH = "data/raw/"
EXTERNAL_DATA_FOLDER_PATH = "src/data/external_data/"
TRANSITIONARY_DATA_FOLDER_PATH = "data/processed/transitionary/"

# PREPROCESSING MOVIES METADATA
def preprocess_movie_metadata():
    df = load_csv(RAW_DATA_FOLDER_PATH + 'movie.metadata.tsv',has_column_names=False, is_tsv=True, column_names=['wikipedia_movie_id', 'freebase_movie_id', 'movie_name', 'movie_release_date',
                            'movie_box_office_revenue', 'movie_runtime', 'movie_languages', 'movie_countries', 'movie_genres'])
    # here, we filter the movies on their date only
    df = to_datetime(df, column='movie_release_date')
    # filtering the years because we have very small data prior to 1910 and 2013
    df = df[(df['movie_release_date'] > 1913) & (df['movie_release_date'] <= 2013)]
    # the data regarding countries, languages and genres are very broad,
    # to avoid getting lost in interpretation we decided to cluster them
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
    
    df.to_csv(TRANSITIONARY_DATA_FOLDER_PATH + "movies_metadata.csv")
    return df


# PREPROCESSING CHARACTER METADATA
def preprocess_char_metadata():
    df = load_csv(RAW_DATA_FOLDER_PATH + 'character.metadata.tsv', has_column_names=False, is_tsv=True, column_names=['wikipedia_movie_id', 'freebase_movie_id', 'movie_release_date', 'char_name',
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

    df.to_csv(TRANSITIONARY_DATA_FOLDER_PATH + "characters_metadata.csv")
    return df


# PREPROCESSING DIRECTOR METADATA
def attribute_gender_to_dir(name_dir, gender_db):
    # attribute gender to director name using the name dictionary
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

def preprocess_movies_director(wiki_movies_id):
    # generation of the dataset movies director using wikipedia webscraping (~ 8h to run!!)
    movies_director = []
    for i in tqdm(wiki_movies_id, desc="Processing movies"):
        infobox_data = get_wikipedia_infobox_by_id(i)
        if (infobox_data is None) or ("Directed by" not in infobox_data):
            continue
    director = infobox_data.get("Directed by")
    movies_director.append({"wikipedia_movie_id": i, "director_name": director})

    movies_director = pd.DataFrame(movies_director)
    movies_director = movies_director.drop_duplicates()
    gender_db = pd.read_csv("src/data/external_data/name_gender_dataset.csv")
    movies_director['director_gender'] = movies_director['director_name'].progress_apply(lambda x: attribute_gender_to_dir(x, gender_db))
    movies_director = movies_director[movies_director['director_gender'].notna()]
    
    movies_director.to_csv(TRANSITIONARY_DATA_FOLDER_PATH + "movies_director.csv")
    return movies_director
 
 
# PREPROCESSING MOVIE SUCCESS
def preprocess_imdb_ratings(movies_metadata_df):
    """ cmu_df is the dataframe containing the movies metadata """
    
    # Load the data
    # the title.basics.tsv and title.ratings.tsv files are too big to be stored in the repository
    
    #imdb_title_df = pd.read_csv("src/data/external/title.basics.tsv", sep='\t')
    #imdb_ratings_df = pd.read_csv("src/data/external_data/raw_data/title.ratings.tsv", sep='\t')
    imdb_title_df = load_csv("src/data/external_data/title.basics.tsv", is_tsv=True)
    imdb_ratings_df = load_csv("src/data/external_data/title.ratings.tsv", is_tsv=True)
    # Merging the two IMDB DataFrames on the 'tconst' column
    imdb_df = pd.merge(imdb_title_df, imdb_ratings_df[['tconst', 'averageRating', 'numVotes']], on='tconst', how='left')
    
    # To be sure to get all names, we consider the two columns primaryTitle and originalTitle
    # We create a match name with the title and the year of the movie, lowercasing the str
    # For IMDB
    imdb_df['primaryTitle_match'] = imdb_df['primaryTitle'].str.lower() + imdb_df['startYear'].str.lower()
    imdb_df['originalTitle_match'] = imdb_df['originalTitle'].str.lower() + imdb_df['startYear'].str.lower()
    imdb_all_titles = pd.concat([imdb_df["primaryTitle_match"], imdb_df["originalTitle_match"]]).drop_duplicates()
    # For CMU
    movies_metadata_titles = movies_metadata_df["movie_name"].str.lower() + movies_metadata_df["movie_release_date"].apply(lambda x: str(x))

    # We get the indexes of the intersection of movies
    intersecting_movies_metadata_titles = movies_metadata_titles.isin(imdb_all_titles)

    # Reshape IMDb data by concatenating both title types into one column along with other columns
    # We drop duplicates on the title
    imdb_ratings_expanded = pd.concat([
        imdb_df[['tconst', 'primaryTitle_match', "startYear", 'averageRating', 'numVotes']].rename(columns={'primaryTitle_match': 'title_match'}),
        imdb_df[['tconst', 'originalTitle_match', "startYear", 'averageRating', 'numVotes']].rename(columns={'originalTitle_match': 'title_match'})
    ]).dropna(subset=['title_match']).drop_duplicates(subset=['title_match'])

    # We keep only the interected df
    movies_metadata_intersection_df = movies_metadata_df.loc[intersecting_movies_metadata_titles]
    # We create the titles match column to merge easily
    movies_metadata_intersection_df['movie_name_match'] = movies_metadata_titles
    
    # We merge
    imdb_in_movies = movies_metadata_intersection_df.merge(
        imdb_ratings_expanded,
        left_on='movie_name_match',
        right_on='title_match',
        how='left'
    )

    # We drop the rows with no ratings
    imdb_in_movies = imdb_in_movies.dropna(subset=["averageRating"])

    # Some movies are duplicated on tconst, we handle them
    # get tconst that are duplicated
    duplicate_tconst_movies = imdb_in_movies.tconst.value_counts().loc[imdb_in_movies.tconst.value_counts()!=1]
    # create a transition df with only duplicated movies
    duplicate_df = imdb_in_movies.loc[imdb_in_movies.tconst.isin(duplicate_tconst_movies.index.values)]
    # count the number of nans on the rows
    duplicate_df["nan_number"] = duplicate_df.isna().sum(axis=1)
    # chose the row where the number of nan is minimum 
    min_nan = duplicate_df.groupby("tconst")["nan_number"].idxmin()
    # take the opposite to get the movies to remove
    duplicate_to_remove = duplicate_df.index.difference(min_nan)
    # remove the duplicates
    imdb_in_movies = imdb_in_movies.drop(duplicate_to_remove, axis=0)
    
    
    # We clean our dataframe to export it
    imdb_in_movies = imdb_in_movies[["wikipedia_movie_id", "tconst", "movie_name", "startYear", "averageRating", "numVotes"]].reset_index(drop=True)
    # We rename the columns
    cols_name = ["imdb_movie_id", "movie_release_date"]
    cols_to_rename = ["tconst", "startYear"]
    rename_mapping = dict(zip(cols_to_rename, cols_name))
    imdb_in_movies.rename(columns=rename_mapping, inplace=True)
    
    return imdb_in_movies
   
def preprocess_movies_success():
    # Loading and preprocessing the data of tmdb (external data)
    tmdb_ratings =  load_csv('src/data/external_data/TMDB_movie_dataset_v11.csv')
    tmdb_ratings = tmdb_ratings[['title', 'vote_average', 'vote_count', 'release_date', 'revenue', 'budget']]
    # We also thought of keeping the popularity column, but we could not retrieve information about the parameters that made a movie "popular" or not, so we decided to drop it.
    # convert 0 values into nan
    tmdb_ratings['revenue'] = tmdb_ratings['revenue'].replace('0', np.nan)
    tmdb_ratings['budget'] = tmdb_ratings['budget'].replace('0', np.nan)
    # We have two different sources (tmdb & imdb) form which we can retrieve the average rating given by people who have watched the movie. In order to decide which dataset we will keep, we will check a specific parameter.
    imdb_ratings = preprocess_imdb_ratings()
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
    movies_success_df = pd.merge(tmdb_ratings, imdb_ratings, on=['movie_name', 'release_date'], how='inner')
    # renaiming the columns
    movies_success_df = movies_success_df[['wikipedia_movie_id', 'revenue', 'budget', 'average_rating', 'num_votes']]
    movies_success_df.rename(columns={'revenue': 'box_office_revenue', 'budget': 'movie_budget'}, inplace=True)
    
    movies_success_df.to_csv(TRANSITIONARY_DATA_FOLDER_PATH + "movies_success.csv")
    return movies_success_df


# PREPROCESSING PLOT EMOTIONS
# Function for analyzing the emotion of a text
def emotion_analysis(text):
    # Load the pipeline with a specific emotion model
    emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

    chunks = [text[i:i+512] for i in range(0, len(text), 512)]
    results = [emotion_pipeline(chunk) for chunk in chunks]

    # Initialize a dictionary to accumulate scores for each emotion
    emotion_scores = defaultdict(list)

    #Process each chunk and accumulate scores
    for chunk_result in results:
        for emotion in chunk_result:
            for i in range(len(emotion)):
                emotion_scores[emotion[i]['label']].append(emotion[i]['score'])
            #print(emotion_scores[emotion['label']])
            #emotion_scores[emotion['label']].append(emotion['score'])

    # Calculate the average score for each emotion
    average_scores = {emotion: sum(scores) / len(scores) for emotion, scores in emotion_scores.items()}

    return average_scores


def preprocess_plot_emotions():
    # Function for processing all film summaries 
    plot_summaries_df = load_csv(RAW_DATA_FOLDER_PATH + 'plot_summaries.txt', is_tsv=True)
    output_file_path = TRANSITIONARY_DATA_FOLDER_PATH + 'plot_emotions.csv'
    batch_size = 100
    for i in range(0, plot_summaries_df.shape[0], batch_size):
        batch = plot_summaries_df.iloc[i:i + batch_size]

        batch['emotion_scores'] = batch['plot'].apply(emotion_analysis)
        batch['dominant_emotion'] = batch['emotion_scores'].apply(lambda x: max(x, key=x.get))

        batch.to_csv(output_file_path, mode='a', index=False, header=not os.path.exists(output_file_path))
        
    return output_file_path # returns the path to the output file


# PREPROCESSING BECHDEL TEST
def preprocess_bechdel_ratings():
    bechdel_ratings_df = 0
    bechdel_ratings_df.to_csv(TRANSITIONARY_DATA_FOLDER_PATH + "bechdel_ratings.csv")  
    return 0


# PREPROCESSING MOVIES COMPLETE
def preprocess_movies_complete(from_files=False):

    if from_files:
        movies_metadata_df = pd.read_csv(TRANSITIONARY_DATA_FOLDER_PATH + "movies_metadata.csv")
        char_metadata_df = pd.read_csv(TRANSITIONARY_DATA_FOLDER_PATH + "characters_metadata.csv") 
        movies_director_df = pd.read_csv(TRANSITIONARY_DATA_FOLDER_PATH + "movies_director.csv")
        movie_success_df = pd.read_csv(TRANSITIONARY_DATA_FOLDER_PATH + "movies_success.csv")
        plot_emotions_df = pd.read_csv(TRANSITIONARY_DATA_FOLDER_PATH + "plot_emotions.csv")
        bechdel_ratings_df = pd.read_csv(TRANSITIONARY_DATA_FOLDER_PATH + "bechdel_ratings.csv")  

    else:
        movies_metadata_df = preprocess_movie_metadata()
        char_metadata_df = preprocess_char_metadata()
    
    # intersection on the first 2 to obtain the same wikipedia_movie_id
    movies_metadata_df = movies_metadata_df[movies_metadata_df['wikipedia_movie_id'].isin(char_metadata_df['wikipedia_movie_id'])]
    char_metadata_df = char_metadata_df[char_metadata_df['wikipedia_movie_id'].isin(movies_metadata_df['wikipedia_movie_id'])]
    
    if from_files == False:
        movies_director_df = preprocess_movies_director(movies_metadata_df['wikipedia_movie_id'].unique())
        movie_success_df = preprocess_movies_success()
        
        plot_emotions_df = load_csv(preprocess_plot_emotions())
        bechdel_ratings_df = preprocess_bechdel_ratings()
    
    # create complete df
    movies_complete_df = movies_metadata_df.copy(deep=True)
    # keep only interesting columns
    movies_complete_df = movies_complete_df[["wikipedia_movie_id", "movie_name", "movie_release_date", "movie_genres", "movie_countries"]]
    movies_complete_df = movies_complete_df.merge(movies_director_df, on="wikipedia_movie_id", how="left")
    # process columns
    movies_complete_df['movie_genres'] = movies_complete_df['movie_genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    movies_complete_df['movie_countries'] = movies_complete_df['movie_countries'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # merging with list of actor genders
    genders_grouped = (
        char_metadata_df.groupby("wikipedia_movie_id")["actor_gender"]
        .apply(list)  # Collect genders into lists
        .reset_index()  # Reset index to make it a DataFrame
        .rename(columns={"actor_gender": "actor_genders"})  # Rename the column
    )
    # merging with list of actor ages
    age_grouped = (
        char_metadata_df.groupby("wikipedia_movie_id")["actor_age"]
        .apply(list)  # Collect genders into lists
        .reset_index()  # Reset index to make it a DataFrame
    )

    movies_complete_df = movies_complete_df.merge(genders_grouped, on="wikipedia_movie_id", how="left")
    movies_complete_df = movies_complete_df.merge(age_grouped, on="wikipedia_movie_id", how="left")

    # merging with bechdel if value exists, else nan
    movies_complete_df = movies_complete_df.merge(bechdel_ratings_df[["wikipedia_movie_id", "bechdel_rating"]], on="wikipedia_movie_id", how="left")
    # merging with success if value exists, else nan
    movies_complete_df = movies_complete_df.merge(movie_success_df[["wikipedia_movie_id", "box_office_revenue", "movie_budget", "average_rating", "num_votes"]], on="wikipedia_movie_id", how="left")
    # merging with plot summaries if value exists, else nan
    movies_complete_df = movies_complete_df.merge(plot_emotions_df[["wikipedia_movie_id", "emotion_scores", "dominant_emotion"]], on="wikipedia_movie_id", how="left")
    # Create new columns for Male and Female character counts
    movies_complete_df["char_M"] = movies_complete_df["actor_genders"].apply(lambda genders: genders.count("M"))
    movies_complete_df["char_F"] = movies_complete_df["actor_genders"].apply(lambda genders: genders.count("F"))
    movies_complete_df["char_tot"] = movies_complete_df["char_M"] + movies_complete_df["char_F"]

    movies_complete_df.to_csv('data/processed/movies_complete.csv', index=False)
    return movies_complete_df
    
    
def preprocessing_bechdel_for_correlation(movies_complete_df):
    df_bechdel = movies_complete_df.copy(deep = True)

    df_bechdel = df_bechdel.dropna(subset=['bechdel_rating', 'emotion_scores'])
    df_bechdel["emotion_scores"] = df_bechdel["emotion_scores"].str.replace("'", '"')
    # Parse the corrected strings into dictionaries
    df_bechdel["emotion_scores"] = df_bechdel["emotion_scores"].apply(json.loads)

    genres_list = df_bechdel.explode('movie_genres')['movie_genres'].unique().tolist()
    countries_list = df_bechdel.explode("movie_countries")["movie_countries"].unique().tolist()
    emotion_list = df_bechdel["dominant_emotion"].unique().tolist()
    # add genre and countries columns
    cols_df = pd.DataFrame(columns= genres_list + countries_list + emotion_list)
    df_bechdel = pd.concat([df_bechdel, cols_df], axis=1).fillna(0).reset_index(drop=True)


    for index, row in df_bechdel.iterrows():
        genres = row["movie_genres"]
        countries = row["movie_countries"]
        emotions_dict = row["emotion_scores"]
        for genre in genres:
            df_bechdel.at[index, genre] = 1
        for country in countries:
            df_bechdel.at[index, country] = 1
        for emotion in emotion_list:
            df_bechdel.at[index, emotion] = emotions_dict[emotion]
    
    # dropping old unformatted columns
    df_bechdel = df_bechdel.drop(columns=["actor_genders", "movie_genres", "movie_countries", "actor_genders", "emotion_scores", "dominant_emotion", "wikipedia_movie_id", "movie_name", "director_name", "actor_age"])
    df_bechdel.columns = df_bechdel.columns.astype(str)

    # simplifying the bechdel_rating column into 0 (fails test) and 1(passes test)
    df_bechdel["bechdel_rating"] = df_bechdel["bechdel_rating"].apply(lambda x: int(0) if (x==0 or x==1 or x==2) else int(1))

    # simplifying the bechdel_rating column into 0 (M) and 1(F)
    df_bechdel["director_gender"] = df_bechdel["director_gender"].apply(lambda x: int(0) if (x=='M') else int(1))

    return df_bechdel


def preprocessing_plot_emotions(movies_complete_df):
    df_plot_emotions = movies_complete_df[['wikipedia_movie_id', 'director_name', 'director_gender', 'emotion_scores', 'dominant_emotion']]
    df_plot_emotions = df_plot_emotions.dropna(subset=['emotion_scores'])
    return df_plot_emotions


def preprocessing_bechdel_for_radar_graph(movies_complete_df):
    df_bechdel = movies_complete_df.copy(deep = True)

    df_bechdel = df_bechdel.dropna(subset=['bechdel_rating', 'emotion_scores'])
    df_bechdel["emotion_scores"] = df_bechdel["emotion_scores"].str.replace("'", '"')
    # Parse the corrected strings into dictionaries
    df_bechdel["emotion_scores"] = df_bechdel["emotion_scores"].apply(json.loads)

    emotion_list = df_bechdel["dominant_emotion"].unique().tolist()
    # add genre and countries columns
    cols_df = pd.DataFrame(columns= emotion_list)
    df_bechdel = pd.concat([df_bechdel, cols_df], axis=1).fillna(0).reset_index(drop=True)


    for index, row in df_bechdel.iterrows():
        emotions_dict = row["emotion_scores"]
        for emotion in emotion_list:
            df_bechdel.at[index, emotion] = emotions_dict[emotion]


    # dropping old unformatted columns
    df_bechdel = df_bechdel.drop(columns=["actor_genders", "movie_genres", "movie_countries", "actor_genders", "emotion_scores", "dominant_emotion", "wikipedia_movie_id", "movie_name", "director_name", "actor_age"])
    df_bechdel.columns = df_bechdel.columns.astype(str)

    # simplifying the bechdel_rating column into 0 (fails test) and 1(passes test)
    df_bechdel["bechdel_rating"] = df_bechdel["bechdel_rating"].apply(lambda x: int(0) if (x==0) else int(3))

    # simplifying the bechdel_rating column into 0 (M) and 1(F)
    df_bechdel["director_gender"] = df_bechdel["director_gender"].apply(lambda x: int(0) if (x=='M') else int(1))
        
    return df_bechdel


def ratio_emotion_by_director_gender(df_plot_emotions):
    df_plot_emotions_women = df_plot_emotions[df_plot_emotions['director_gender'] == 'F']
    df_plot_emotions_men = df_plot_emotions[df_plot_emotions['director_gender'] == 'M']

    emotion_totals_women = {
        'anger': 0,
        'disgust': 0,
        'fear': 0,
        'joy': 0,
        'neutral': 0,
        'sadness': 0,
        'surprise': 0
    }
    emotion_totals_men = {
        'anger': 0,
        'disgust': 0,
        'fear': 0,
        'joy': 0,
        'neutral': 0,
        'sadness': 0,
        'surprise': 0
    }

    for emotion_score in df_plot_emotions_women['emotion_scores']:
        scores = ast.literal_eval(emotion_score)
        for emotion, score in scores.items():
            emotion_totals_women[emotion] += score

    for emotion_score in df_plot_emotions_men['emotion_scores']:
        scores = ast.literal_eval(emotion_score)
        for emotion, score in scores.items():
            emotion_totals_men[emotion] += score


    # Calculating ratios
    total_women = sum(emotion_totals_women.values())
    total_men = sum(emotion_totals_men.values())

    ratios_women = {emotion: value / total_women * 100 for emotion, value in emotion_totals_women.items()}
    ratios_men = {emotion: value / total_men * 100 for emotion, value in emotion_totals_men.items()}

    return ratios_women, ratios_men


    
# PREPROCESSING TV TROPES
def transform_title(title):
    # Remove non-alphanumeric characters except spaces
    title = ''.join(char if char.isalnum() or char.isspace() else '' for char in title)
    # Convert to PascalCase
    return ''.join(word.capitalize() for word in title.split())

def preprocess_tvtropes(df):
    # Normalize character types
    tvtropes_df = load_csv(RAW_DATA_FOLDER_PATH + 'tvtropes.clusters.txt', has_column_names=False, is_tsv=True, column_names=['character_type', 'metadata'])
    film_tropes_df = load_csv(EXTERNAL_DATA_FOLDER_PATH + "film_tropes.csv")
    genderedness_df = load_csv(EXTERNAL_DATA_FOLDER_PATH + "genderedness_filtered.csv")

    char_types = [''.join(word.capitalize() for word in trope.split('_')) for trope in tvtropes_df['character_type'].unique()]

    # Filter film tropes based on character types and transformed movie titles
    valid_titles = [transform_title(name) for name in df['movie_name'].unique()]
    filtered_tropes = film_tropes_df[
        film_tropes_df['Trope'].isin(char_types) & film_tropes_df['Title'].isin(valid_titles)
    ].drop(['title_id', 'Unnamed: 0'], axis=1)

    # Merge with transformed movie names
    movies_data = df.copy()
    movies_data['movie_name'] = movies_data['movie_name'].apply(transform_title)
    filtered_tropes = filtered_tropes.join(
        movies_data.set_index('movie_name'), on='Title'
    ).dropna()

    return filtered_tropes, genderedness_df

