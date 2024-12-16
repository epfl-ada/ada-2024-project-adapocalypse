from collections import defaultdict

def process_bechdel_ratings(ratings):
    """
    Process the Bechdel ratings to a more readable format.
    """
    if ratings == 0:
        return "No information"
    elif ratings == 1:
        return "Not passing"
    elif ratings == 2:
        return "Passing"
    elif ratings == 3:
        return "Passing with caveats"
    else:
        return "Invalid rating"

# Function for analyzing the emotion of a text
def emotion_analysis(df, text):
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

# get the emotions and dominant emotion for the first n plots
def n_first_emotions(n, init_df):
    df = init_df.head(n)
    df['emotion_scores'] = df['plot_summary'].progress_apply(emotion_analysis)
    df['dominant_emotion'] = df['emotion_scores'].apply(lambda x: max(x, key=x.get))
    return df

def actor_age_according_to_dir_gender(df, gender):
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

def group_formation(df, opti):
    actor_gender_movie_data = df.groupby('wikipedia_movie_id')['actor_gender'].value_counts()
    prop_female_actors = actor_gender_movie_data[:, 'F'] / actor_gender_movie_data.groupby(level=0).sum()
    if (opti):
        bechdel_wiki_movie_id = df[df['bechdel_rating'] == 3]['wikipedia_movie_id']
        fem_rep_wiki_movie_id = prop_female_actors[prop_female_actors > 0.35].index
    else:
        bechdel_wiki_movie_id = df[df['bechdel_rating'] <= 2]['wikipedia_movie_id']
        fem_rep_wiki_movie_id = prop_female_actors[prop_female_actors <= 0.35].index
    optimal_df = df.loc[df['wikipedia_movie_id'].isin(set(bechdel_wiki_movie_id).intersection(set(fem_rep_wiki_movie_id)))]
    return optimal_df