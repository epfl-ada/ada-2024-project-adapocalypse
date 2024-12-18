from collections import defaultdict, Counter
import pandas as pd
import numpy as np
import json

from src.data.data_loader import load_csv

# IMPORTATIONS FOR THE ML MODEL
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from statsmodels import tools
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import ast

# CONSTANT DEFINITIONS
RAW_DATA_FOLDER_PATH = 'data/raw/'
EXTERNAL_DATA_FOLDER_PATH = 'src/data/external_data/'

# FUNCTIONS USED IN THE PREPROCESSING OF THE DATA
def emotion_analysis(df, text):
    """ Calculate the average score of each emotion (anger, disgust, joy, fear, neutral, sadness, surprise) in a text

    Args:
        df (DataFrame): original dataframe on which we perform the analysis
        text : text on which to perform emotion analysis (in our case the plot summaries)

    Returns:
        dict: dictionary attributing to each emotion its average score
    """
    chunks = [text[i:i+512] for i in range(0, len(text), 512)]
    results = [df(chunk) for chunk in chunks]

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

# FUNCTIONS USED TO PROCESS DATA TO ADAPT IT SPECIFICALLY TO GRAPHS

# 2.
def process_movies_by_country(df):
    """Computes the number of movies produced by each country and filters out the countries with less than 500 movies

    Args:
        df (DataFrame): original dataframe that is to be modified to produce the desired data

    Returns:
        DataFrame: filtered and modified dataframe, which data is ready to be plotted
    """
    all_countries = df['movie_countries'].copy(deep=True)
    all_countries = [genre for sublist in df['movie_countries'] for genre in sublist]
    countries_counts = Counter(all_countries)

    country_df = pd.DataFrame(countries_counts.items(), columns=['country', 'number_of_movies'])
    country_df = country_df.sort_values(by='number_of_movies', ascending=False)
    country_df = country_df[country_df['number_of_movies'] > 500]
    return country_df

# 2.
def process_movies_by_genre(df):
    """Computes the number of movies produced in each movie genre and filters out the genres with less than 500 movies

    Args:
        df (DataFrame): original dataframe that is to be modified to produce the desired data

    Returns:
        DataFrame: filtered and modified dataframe, which data is ready to be plotted
    """
    all_genres = df['movie_genres'].copy()
    all_genres = [genre for sublist in df['movie_genres'] for genre in sublist]
    genre_counts = Counter(all_genres)
    genre_df = pd.DataFrame(genre_counts.items(), columns=['movie_genre', 'number_of_movies'])

    genre_df = genre_df.sort_values(by='number_of_movies', ascending=False)
    # keep only the genres with more than 500 movies
    genre_df = genre_df[genre_df['number_of_movies'] > 500]
    return genre_df

# 3.B 2)
def process_actor_age(df, gender):
    """Computes the age distribution (in percentage) of the actors depending on the gender of their Movie Director

    Args:
        df (DataFrame): original dataframe that is to be modified to produce the desired data
        gender (str): 'F' if Female, 'M' if Male

    Returns:
        float, float: age distribution of female / male actors in percentage (playing in movies directed by a director of the specified gender)
    """
    char_gender_directed = df[df['director_gender'] == gender]

    # Filter and clean data
    char_gender_directed_age = char_gender_directed[
        ((~char_gender_directed['actor_age'].isnull()) & 
         (char_gender_directed['actor_age'] < 100) & 
         (char_gender_directed['actor_age'] > 0))
    ]

    # Prepare normalized percentage data for directors of directors gender
    age_fem_actors = char_gender_directed_age[char_gender_directed_age['actor_genders'] == 'F']['actor_age']
    age_male_actors = char_gender_directed_age[char_gender_directed_age['actor_genders'] == 'M']['actor_age']

    age_female_percentage = age_fem_actors.value_counts(normalize=True).sort_index() * 100
    age_male_percentage = age_male_actors.value_counts(normalize=True).sort_index() * 100

    return age_female_percentage, age_male_percentage

# 3.C 1)
def process_bechdel(df):
    """
    Process dataframe to later on apply correlation analysis

    Args:
        df (DataFrame): original dataframe that is to be modified to produce the desired data

    Returns:
        DataFrame: desired dataframe to apply correlation analysis
    """
    df_bechdel = df.copy(deep = True)

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

# 3.C 2)
def process_emotions(df):
    """
    Process dataframe to later on plot radar chat on emotions as well as the distribution of emotions.

    Args: 
        df (DataFrame): original dataframe that is to be modified to produce the desired data

    Returns: 
        DataFrame: desired dataframe to plot the graphs.
    """
    df_plot_emotions = df[['wikipedia_movie_id', 'director_name', 'director_gender', 'emotion_scores', 'dominant_emotion']]
    df_plot_emotions = df_plot_emotions.dropna(subset=['emotion_scores'])
    return df_plot_emotions

# 3.C 2)
def process_emotion_by_dir_gender(df):
    """
    Process dataframe to later on plot the ratio of emotions by gender of the movie director.

    Args: 
        df (DataFrame): original dataframe that is to be modified to produce the desired data

    Returns: 
        DataFrame: desired dataframe to plot the graphs.
    """
    df_plot_emotions_women = df[df['director_gender'] == 'F']
    df_plot_emotions_men = df[df['director_gender'] == 'M']

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

# 3.C 2)
def process_bechdel_radar(df):
    """
    Process dataframe to later on plot radar chat on emotions

    Args:
        df (DataFrame): original dataframe that is to be modified to produce the desired data

    Returns:
        DataFrame: desired dataframe to apply correlation analysis
    """
    df_bechdel = df.copy(deep = True)

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



# 3.C 3)
def logistic_regression_for_bechdel(df):
    """
    Machine Learning logistic regression model to predict the Bechdel test rating of a movie based on its features

    Args:
        df (DataFrame): Processed dataframe on which we extract the data from

    Returns:
        type1, type2, type3, type4:?? Mahlia aide-moi ma soeur
    """
    df_bechdel = df.copy(deep = True)

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
    

    # preparing data for model
    target_cols = ["bechdel_rating"]
    features_cols = df_bechdel.keys().tolist()
    features_cols.remove(target_cols[0])
    X = df_bechdel[features_cols]
    y = df_bechdel[target_cols]

    # we split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # we squeeze to match the dimensions
    y_test = y_test.squeeze()
    y_train = y_train.squeeze()

    # Load the module
    scaler = StandardScaler()

    # Standardize X train
    scaler.fit(X_train) # used both for X_train and X_test
    X_train_standardized = scaler.transform(X_train)
    X_test_standardized = scaler.transform(X_test)

    # we create the model
    log_reg_model = LogisticRegression(random_state=42, max_iter=500)

    # Fit the model
    log_reg_model.fit(X_train_standardized, y_train)

    y_pred_test = log_reg_model.predict(X_test_standardized)
    y_pred_train = log_reg_model.predict(X_train_standardized)

    print(f'The accuracy score for test set is: {accuracy_score(y_test, y_pred_test)}')
    print(f'The accuracy score for train set is: {accuracy_score(y_train, y_pred_train)}')

    return y_test, y_pred_test, log_reg_model, X_train


def preprocessing_bechdel_for_radar_graph(movies_complete_df):
    """
    Process dataframe to later on plot radar chat on emotions regarding the bechdel rating

    Args:
        movies_complete_df (DataFrame): original dataframe that is to be modified to produce the desired data

    Returns:
        DataFrame: desired dataframe needed for plotting radar chat on emotions regarding the bechdel rating
    
    """
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
    df_bechdel["bechdel_rating"] = df_bechdel["bechdel_rating"].apply(lambda x: int(0) if (x==0 or x==1 or x==2) else int(1))

    # simplifying the bechdel_rating column into 0 (M) and 1(F)
    df_bechdel["director_gender"] = df_bechdel["director_gender"].apply(lambda x: int(0) if (x=='M') else int(1))
        
    return df_bechdel

# 3.D
def get_dominant_tropes(filtered_tropes, genderedness_df, dominant, non_dominant):
    """Determines the distribution of gendered final tropes regarding the gender of the movie director

    Args:
        filtered_tropes (DataFrame): contains the tropes filtered
        genderedness_df (DataFrame): contains information regarding the genderedness of the tropes
        dominant (str): "MaleTokens" or "FemaleTokens" depending on the gender we want to examine
        non_dominant (_type_): "MaleTokens" or "FemaleTokens" depending on the gender we do not want to examine

    Returns:
        DataFrame: Male/Female tropes most represented, in descending order of appearances
    """
    # Filter genderedness stats for tropes and female-dominant ones
    gendered_tropes = genderedness_df[genderedness_df['Trope'].isin(filtered_tropes['Trope'].unique())]
    dominant_tropes = gendered_tropes[gendered_tropes[dominant] > gendered_tropes[non_dominant]]
    # Sort the filtered data by 'TotalMFTokens' in descending order and take the top 6
    dominant_tropes = dominant_tropes.sort_values(by=dominant, ascending=False)

    # Aggregate data for female-dominant tropes
    trope_list = dominant_tropes['Trope'].tolist()
    filtered_tropes = filtered_tropes[filtered_tropes['Trope'].isin(trope_list)]
    aggregated_data = filtered_tropes.groupby('Trope').agg({
        'director_gender': list,
        'Title': list,
        'movie_genres': list
    })
    final_tropes = dominant_tropes.merge(aggregated_data, on='Trope', how='left')

    # Add columns for director counts and percentages
    final_tropes = final_tropes.assign(trope_gender=0, dir_M=0, dir_F=0, dir_M_perc=0, dir_F_perc=0)

    final_tropes["trope_gender"] = "Male tropes" if dominant=="MaleTokens" else "Female tropes"

    # Count male and female directors
    final_tropes[['dir_M', 'dir_F']] = final_tropes['director_gender'].apply(
        lambda genders: pd.Series([genders.count('M'), genders.count('F')])
    )

    return final_tropes

# 3.D
def preprocessing_final_tropes(final_tropes_F, final_tropes_M):
    """Processes data to plot the gendered proportion of tv tropes depending on the gender of the movie director

    Args:
        final_tropes_F (DataFrame): Final Tropes that describe female characters, sorted in descending order of female appearances
        final_tropes_M (DataFrame): Final Tropes that describe male characters, sorted in descending order of male appearances

    Returns:
        tuple of floats, tuple of floats: proportion of male/female tropes in movies depending on the gender of the movie director
    """
    gendered_tropes = pd.concat([final_tropes_F, final_tropes_M])
    # Total male and female directors
    total_M_dir = gendered_tropes['dir_M'].sum()
    total_F_dir = gendered_tropes['dir_F'].sum()

    # Calculate percentages
    gendered_tropes['dir_M_perc'] = gendered_tropes['dir_M'] / total_M_dir * 100 if total_M_dir > 0 else 0
    gendered_tropes['dir_F_perc'] = gendered_tropes['dir_F'] / total_F_dir * 100 if total_F_dir > 0 else 0

    male_director_data = gendered_tropes.groupby('trope_gender')['dir_M_perc'].sum()
    female_director_data = gendered_tropes.groupby('trope_gender')['dir_F_perc'].sum()  
    
    return male_director_data, female_director_data

# 3.E 1)
def group_formation(df, opti):
    """Proceeds to form the "Optimal" and "Worst" group based on the selection criterias listed in the notebook

    Args:
        df (DataFrame): original dataframe that is to be modified to produce the desired data
        opti (bool): "True" if we want to form the "Optimal" group, "False" if we want to form the "Worst" group

    Returns:
        DataFrame: dataframe that fits the conditions of its group
    """
    prop_female_actors = df['char_F'] / df['char_tot']
    prop_female_actors.index = df['wikipedia_movie_id'].copy(deep=True)
    if (opti):
        bechdel_wiki_movie_id = df[df['bechdel_rating'] == 3]['wikipedia_movie_id']
        fem_rep_wiki_movie_id = prop_female_actors[prop_female_actors > 0.35].index
    else:
        bechdel_wiki_movie_id = df[df['bechdel_rating'] <= 2]['wikipedia_movie_id']
        fem_rep_wiki_movie_id = prop_female_actors[prop_female_actors <= 0.35].index
    optimal_df = df.loc[df['wikipedia_movie_id'].isin(set(bechdel_wiki_movie_id).intersection(set(fem_rep_wiki_movie_id)))]
    return optimal_df