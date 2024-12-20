from collections import defaultdict, Counter
import pandas as pd
import numpy as np
import json

from src.data.data_loader import load_csv

# IMPORTATIONS FOR THE ML MODEL AND THE STATISTICAL TESTS
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
from scipy.stats import chi2_contingency
import scipy.stats as stats
import ast
import joblib


# CONSTANT DEFINITIONS
RAW_DATA_FOLDER_PATH = 'data/raw/'
EXTERNAL_DATA_FOLDER_PATH = 'src/data/external_data/'

# FUNCTIONS USED IN THE PREPROCESSING OF THE DATA
def emotion_analysis(df, text):
    """ 
    Calculate the average score of each emotion (anger, disgust, joy, fear, neutral, sadness, surprise) in a text

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
    """
    Computes the number of movies produced by each country and filters out the countries with less than 500 movies

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
    """
    Computes the number of movies produced in each movie genre and filters out the genres with less than 500 movies

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
    """
    Computes the age distribution (in percentage) of the actors depending on the gender of their Movie Director

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

# 3.B 4)
def process_map_fem_char(movies_df, director_gender, filter_movies):
    """


    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    # Filter the data for the specified director gender
    movies_df = movies_df[movies_df["director_gender"] == director_gender]

    # Flatten the "movie_countries" column to the first country
    movies_df["movie_countries"] = movies_df["movie_countries"].str[0]

    # Compute percentage of female characters for each movie
    movies_df["female_percentage"] = (movies_df["char_F"] / movies_df["char_tot"]) * 100

    # Group by country to calculate metrics
    number_movies_per_country = movies_df.groupby("movie_countries")["wikipedia_movie_id"].nunique()
    countries_filter_movies = (number_movies_per_country.values > filter_movies)
    mean_female_percentage_per_country = movies_df.groupby("movie_countries")["female_percentage"].mean()
    mean_female_percentage_per_country = mean_female_percentage_per_country[countries_filter_movies]
    total_characters_per_country = movies_df.groupby("movie_countries")["char_tot"].sum()
    total_characters_per_country = total_characters_per_country[countries_filter_movies]

    # Create a DataFrame with the results
    gender_percentages_per_country = pd.DataFrame({
        "mean_female_percentage": mean_female_percentage_per_country,
        "movies_count": number_movies_per_country,
        "total_characters": total_characters_per_country
    }).reset_index()

    # Sort data for better readability (optional)
    gender_percentages_per_country = gender_percentages_per_country.sort_values(by="mean_female_percentage", ascending=False)
    return gender_percentages_per_country

# 3.B 5)
def process_top10_genres(df):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    genre_gender_counts = df.explode("actor_genders").explode("movie_genres")
    gender_genre_counts = genre_gender_counts.groupby(['movie_genres', 'actor_genders']).size().unstack(fill_value=0)
    gender_genre_percentages = gender_genre_counts.div(gender_genre_counts.sum(axis=1), axis=0) * 100
    genre_counts = genre_gender_counts.groupby('movie_genres').size().sort_values(ascending=False).head(10)
    gender_genre_percentages_top10 = gender_genre_percentages.loc[genre_counts.index]
    return gender_genre_percentages_top10


# 3.C
def get_dominant_tropes(filtered_tropes, genderedness_df, dominant, non_dominant):
    """
    Determines the distribution of gendered final tropes regarding the gender of the movie director

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


# 3.C
def get_top_tropes(trope_data, perc_column):
    """ Function to limit to get the top-tropes
    """
    # Sort the tropes based on percentage
    top_tropes = trope_data.sort_values(by=perc_column, ascending=False)
    return top_tropes

def preprocessing_final_tropes(final_tropes_F, final_tropes_M):
    """
    Processes data to plot the gendered proportion of tv tropes depending on the gender of the movie director

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

    # Get top tropes for Male and Female directors
    male_director_top = get_top_tropes(gendered_tropes, 'dir_M_perc')
    female_director_top = get_top_tropes(gendered_tropes, 'dir_F_perc')
    
    return male_director_data, female_director_data, male_director_top, female_director_top


# 3.C.2 )
def top_genres(df, big_cols, less_cols, top=3):  
    # Group by then count the occurrences
    genre_counts = df.explode("movie_genres").groupby(big_cols)["movie_genres"].count().reset_index(name='count')
    # Sort the values by count in descending order
    sorted_genre_counts = genre_counts.sort_values(by='count', ascending=False)
    # Get top genres
    top_genres_per_trope = sorted_genre_counts.groupby(less_cols).head(top)
    top_genres_per_trope = top_genres_per_trope.groupby(less_cols)['movie_genres'].apply(list).reset_index()
    
    return top_genres_per_trope

# 3.D 1)
def corr_bechdel(df):
    """
    Perform Pearson's correlation test for the number of male and female characters with Bechdel ratings.
    Also performs the Chi-Square test for director gender and Bechdel rating.
    
    Args:
        df (DataFrame): Processed dataframe on which we extract the data from
    """
    # Correlation values (Pearson's correlation for numerical features)
    correlations = df[['bechdel_rating', 'char_F', 'char_M']].corr()
    bechdel_corr = correlations['bechdel_rating'].drop('bechdel_rating')

    # Create an empty dictionary for p-values
    p_values = {}

    # Perform Pearson's correlation for character counts (char_F, char_M)
    for col in ['char_F', 'char_M']:
        corr, p_val = stats.pearsonr(df[col], df['bechdel_rating'])
        p_values[col] = p_val

    # Perform Chi-Square test for director_gender and bechdel_rating
    contingency_table = pd.crosstab(df['director_gender'], df['bechdel_rating'])
    chi2_stat, p_val_chi2, dof, expected = stats.chi2_contingency(contingency_table)
    chi2_results = {
        'Chi-Square Statistic': chi2_stat,
        'P-Value': p_val_chi2,
        'Degrees of Freedom': dof,
        'Expected Frequencies': expected,
        'Observed Frequencies': contingency_table.values
    }

    # Combine correlations and p-values into one DataFrame for easy display
    corr_results = pd.DataFrame({
        'Correlation': bechdel_corr,
        'P-Value': p_values
    })

    # Print the correlation results and significance
    print("Correlation Results with Bechdel Rating:")
    for index, row in corr_results.iterrows():
        if row['P-Value'] < 0.05:
            print(f"Variable: {index}, Correlation: {row['Correlation']:.2f}, P-Value: {row['P-Value']} (Significant)")

        else:
            print(f"Variable: {index}, Correlation: {row['Correlation']:.2f}, P-Value: {row['P-Value']} (Not Significant)")

    # Print Chi-Square test results for director_gender and bechdel_rating
    print("\nChi-Square Test for Director Gender and Bechdel Rating:")
    print(f"Chi-Square Statistic: {chi2_results['Chi-Square Statistic']:.2f}")
    print(f"P-Value: {chi2_results['P-Value']}")
    print(f"Degrees of Freedom: {chi2_results['Degrees of Freedom']}")
    print("Observed Frequencies:\n", chi2_results['Observed Frequencies'])
    print("Expected Frequencies:\n", chi2_results['Expected Frequencies'])


def chi2_test(df):
    # Create a contingency table (cross-tabulation of director_gender and bechdel_rating)
    contingency_table = pd.crosstab(df['director_gender'], df['bechdel_rating'], margins=False)

    # Perform the Chi-Square test
    chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)

    # Print the observed frequencies
    print("Observed Frequencies:")
    print(np.array(contingency_table))

    # Print the expected frequencies
    print("\nExpected Frequencies:")
    print(np.array(pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)))

    # Print the Chi-Square statistic, p-value, degrees of freedom, and expected frequencies
    print("\nChi-Square Statistic:", chi2_stat)
    print("Degrees of Freedom:", dof)
    print("P-Value:", p_val)

# 3.D 2)
def process_df_emotions(df):
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

# 3.D 2)
def process_emotion_scores(df):
    """
    Process emotion scores to later on plot its distribution

    Args:
        df (DataFrame): original dataframe that is to be modified to produce the desired data

    Returns:
        dict: dictionary containing the total score for each emotion
    """
    emotion_totals = {
        'anger': 0,
        'disgust': 0,
        'fear': 0,
        'joy': 0,
        'neutral': 0,
        'sadness': 0,
        'surprise': 0
    }
    
    for emotion_score in df['emotion_scores']:
        scores = ast.literal_eval(emotion_score)
        for emotion, score in scores.items():
            emotion_totals[emotion] += score
            
    return emotion_totals

# 3.D 2)
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

# 3.D 2)
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

# 3.D 3)
def process_bechdel(bechdel_df):
    """
    Process dataframe to later on apply correlation analysis

    Args:
        df (DataFrame): original dataframe that is to be modified to produce the desired data

    Returns:
        DataFrame: desired dataframe to apply correlation analysis
    """
    bechdel_df = bechdel_df.dropna(subset=['bechdel_rating', 'emotion_scores'])
    bechdel_df["emotion_scores"] = bechdel_df["emotion_scores"].str.replace("'", '"')
    # Parse the corrected strings into dictionaries
    bechdel_df["emotion_scores"] = bechdel_df["emotion_scores"].apply(json.loads)

    genres_list = bechdel_df.explode('movie_genres')['movie_genres'].unique().tolist()
    countries_list = bechdel_df.explode("movie_countries")["movie_countries"].unique().tolist()
    emotion_list = bechdel_df["dominant_emotion"].unique().tolist()
    # manual one-hot encoding
    cols_df = pd.DataFrame(columns= genres_list + countries_list + emotion_list) # create cols
    bechdel_df = pd.concat([bechdel_df, cols_df], axis=1).fillna(0).reset_index(drop=True) # add empty cols
    for index, row in bechdel_df.iterrows(): # fill the cols
        genres = row["movie_genres"]
        countries = row["movie_countries"]
        emotions_dict = row["emotion_scores"]
        for genre in genres:
            bechdel_df.at[index, genre] = 1
        for country in countries:
            bechdel_df.at[index, country] = 1
        for emotion in emotion_list:
            bechdel_df.at[index, emotion] = emotions_dict[emotion]
    # dropping old unformatted columns
    bechdel_df = bechdel_df.drop(columns=["actor_genders", "movie_genres", "movie_countries", "actor_genders", "emotion_scores", "dominant_emotion", "wikipedia_movie_id", "movie_name", "director_name", "actor_age"])
    bechdel_df.columns = bechdel_df.columns.astype(str)
    # simplifying the bechdel_rating column into 0 (fails test) and 1(passes test)
    bechdel_df["bechdel_rating"] = bechdel_df["bechdel_rating"].apply(lambda x: int(0) if (x==0 or x==1 or x==2) else int(1))
    # simplifying the bechdel_rating column into 0 (M=male director) and 1(F=female director)
    bechdel_df["director_gender"] = bechdel_df["director_gender"].apply(lambda x: int(0) if (x=='M') else int(1))
    
    return bechdel_df

def logistic_regression_for_bechdel(df):
    """
    Machine Learning logistic regression model to predict the Bechdel test rating of a movie based on its features.
    We split 80:20. We standardize then we train the model and we evaluate it. We save it for easy re-use.

    Args:
        df (DataFrame): Processed dataframe on which we extract the data from

    Returns:
        y_test: the test labels.
        y_pred_test: the predicted labels to be compared to y_test.
        log_reg_model: our model.
        X_train: the formatted dataset used to train.
    """
    # call function for pre-processing
    bechdel_df = process_bechdel(df)

    # PREPARING DATA FOR MODEL
    target_cols = ["bechdel_rating"]
    features_cols = bechdel_df.keys().tolist()
    features_cols.remove(target_cols[0])
    X = bechdel_df[features_cols]
    y = bechdel_df[target_cols]

    # we split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # we squeeze to match the dimensions
    y_test = y_test.squeeze()
    y_train = y_train.squeeze()

    # Load the module to standaradize
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

    # Save the model and the scaler
    joblib.dump(log_reg_model, 'src/utils/model_ML/log_reg_model.pkl')
    joblib.dump(scaler, 'src/utils/model_ML/scaler.pkl')

    print(f'The accuracy score for the TEST set is: {accuracy_score(y_test, y_pred_test) * 100:.2f}%')
    print(f'The accuracy score for the TRAINING set is: {accuracy_score(y_train, y_pred_train) * 100:.2f}%')

    return y_test, y_pred_test, log_reg_model, X_train

def obtain_df_bechdel_used_in_ML(df):
    """
    First step of the ML model, we process the dataframe to later on apply the logistic regression model

    Args:
        df (DataFrame): Processed dataframe on which we extract the data from

    Returns:
        df (DataFrame): Processed dataframe to apply the logistic regression model
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


def preprocess_before_inference(df): 
    """
    Preprocess the data for use in our ML model 

    Args: 
        df (DataFrame): Dataframe on which we extract the data for the preprocessing
    Returns:
        df_preprocessed (DataFrame): preprocessed dataframe

    """
    df_without_emotions = df.copy(deep = True)
    df_full_emotions = df.copy(deep = True)

    df_without_emotions = df_without_emotions[df_without_emotions['bechdel_rating'].isna()]

    genres_list = df_without_emotions.explode('movie_genres')['movie_genres'].unique().tolist()
    countries_list = df_without_emotions.explode("movie_countries")["movie_countries"].unique().tolist()

    cols_df_genres_countries = pd.DataFrame(columns= genres_list + countries_list)
    df_without_emotions = pd.concat([df_without_emotions, cols_df_genres_countries], axis=1).fillna(0).reset_index(drop=True)

    for index, row in df_without_emotions.iterrows():
        genres = row["movie_genres"]
        countries = row["movie_countries"]
        for genre in genres:
            df_without_emotions.at[index, genre] = 1
        for country in countries:
            df_without_emotions.at[index, country] = 1
        
    # dropping old unformatted columns
    df_without_emotions = df_without_emotions.drop(columns=["actor_genders", "movie_genres", "movie_countries", "actor_genders", "emotion_scores", "dominant_emotion", "wikipedia_movie_id", "movie_name", "director_name", "actor_age"])
    df_without_emotions.columns = df_without_emotions.columns.astype(str)

    df_without_emotions["director_gender"] = df_without_emotions["director_gender"].apply(lambda x: int(0) if (x=='M') else int(1))


    
    df_full_emotions = df_full_emotions[df_full_emotions['bechdel_rating'].isna()]
    df_full_emotions = df_full_emotions.dropna(subset=['emotion_scores'])

    df_full_emotions["emotion_scores"] = df_full_emotions["emotion_scores"].str.replace("'", '"')
    # Parse the corrected strings into dictionaries
    df_full_emotions["emotion_scores"] = df_full_emotions["emotion_scores"].apply(json.loads)

    emotion_list = df_full_emotions["dominant_emotion"].unique().tolist()
    # add genre and countries columns
    cols_df = pd.DataFrame(columns= emotion_list) # genres_list + countries_list + 
    df_full_emotions = pd.concat([df_full_emotions, cols_df], axis=1).fillna(0).reset_index(drop=True)


    for index, row in df_full_emotions.iterrows():
        emotions_dict = row["emotion_scores"]
        for emotion in emotion_list:
            df_full_emotions.at[index, emotion] = emotions_dict[emotion]


    # dropping old unformatted columns
    df_full_emotions = df_full_emotions.drop(columns=["actor_genders", "movie_genres", "movie_countries", "actor_genders", "emotion_scores", "dominant_emotion", "wikipedia_movie_id", "movie_name", "director_name", "actor_age"])
    df_full_emotions.columns = df_full_emotions.columns.astype(str)

    # simplifying the bechdel_rating column into 0 (M) and 1(F)
    df_full_emotions["director_gender"] = df_full_emotions["director_gender"].apply(lambda x: int(0) if (x=='M') else int(1))

    complicated_columns = ['movie_budget', 'bechdel_rating', 'char_M', 'movie_release_date',
       'num_votes', 'char_F', 'director_gender', 'box_office_revenue',
       'average_rating', 'char_tot']
    
    df_full_emotions = df_full_emotions.drop(columns=complicated_columns)

    df_preprocessed = pd.concat([df_without_emotions, df_full_emotions], axis=1, join="outer")


    df_bechdel = obtain_df_bechdel_used_in_ML(df)
    column_bechdel = list((set(df_bechdel.columns)))
    df_preprocessed['Qatar'] = 0
    df_preprocessed = df_preprocessed[column_bechdel]
    # reorder columns
    df_preprocessed = df_preprocessed[df_bechdel.columns]
    # we don't need the bechdel_rating column for the inference
    df_preprocessed = df_preprocessed.drop(columns=['bechdel_rating'])

    
    return df_preprocessed


def obtain_prediction_bechdel(df_preprocessed):
    """
        Obtain the bechdel prediction for the movies that we don't have the bechdel rating for

        Args: 
            df_preprocessed (DataFrame): Dataframe on which we extract the data for the prediction
        Returns:
            df_bechdel_predictions (DataFrame): prediction dataframe with only the bechdel rating column

    """

    log_reg_model = joblib.load('log_reg_model.pkl')
    scaler = joblib.load('scaler.pkl')

    standardized_preprocessed = scaler.transform(df_preprocessed)
    standardized_preprocessed = np.nan_to_num(standardized_preprocessed)

    predictions_bechdel = log_reg_model.predict(standardized_preprocessed)

    df_bechdel_predictions = pd.DataFrame(predictions_bechdel, columns=['bechdel_rating'])

    return df_bechdel_predictions


def preprocess_bechdel_ratings_by_dirctor_gender(df_preprocessed):
    """
    Preprocess the data for plot the bechdel results by director gender
    Args:
        df_preprocessed (DataFrame): Dataframe on which we extract the data for the preprocessing
    Returns:
        df_plot (DataFrame): preprocessed dataframe for the plot
    """
    df_plot = pd.DataFrame()
    df_bechdel_predictions = obtain_prediction_bechdel(df_preprocessed)
    df_plot['bechdel_rating'] = df_bechdel_predictions['bechdel_rating']
    df_plot['director_gender'] = df_preprocessed['director_gender']
    return df_plot

# 3.C 3)
def feature_importance(log_reg_model, X_train):
    """
    Defines the feature importance of the logistic regression model

    Args:
        log_reg_model : logistic regression model
        X_train (Series): data we want to train

    Returns:
        DataFrame: feature, ranked by descending order
    """
    # Get the feature importance
    selected_features = X_train.columns

    # Get the coefficients from the logistic regression model
    coefficients = log_reg_model.coef_[0]  # For binary classification

    # Create a DataFrame for feature importance
    feature_importance_df = pd.DataFrame({
        'Feature': selected_features,
        'Coefficient': coefficients,
        'Importance': abs(coefficients)
    }).sort_values(by='Importance', ascending=False)

    # Display the top 20 features
    return feature_importance_df.head(10)

# 3.D 3)
def preprocessing_bechdel_for_radar_graph(movies_complete_df):
    """
    Process dataframe to later on plot radar chat on emotions regarding the bechdel rating

    Args:
        movies_complete_df (DataFrame): Processed dataframe on which we extract the data from

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

# 3.E 1)
def process_revenue_bechdel(df):
    """
    Process bechdel dataframe to only keep 'box_office_revenue' values

    Args:
        df (DataFrame): Processed dataframe on which we extract the data from

    Returns:
        Serie, Serie: filtered series containing the adequate values
    """
    df = df[df['box_office_revenue'] < 200000000] # filter outliers to improve readbility

    male_df = df[df['director_gender'] == 'M']['box_office_revenue']
    female_df = df[df['director_gender'] == 'F']['box_office_revenue']
    
    return male_df, female_df

# 3.E 1)
def process_ratings_bechdel(df):
    """
    Process bechdel dataframe to only keep 'average_rating' values

    Args:
        df (DataFrame): Processed dataframe on which we extract the data from

    Returns:
        Serie, Serie: filtered series containing the adequate values
    """
    male_df = df[df['director_gender'] == 'M']['average_rating']
    female_df = df[df['director_gender'] == 'F']['average_rating']
    
    return male_df, female_df

