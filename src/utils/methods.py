import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup


def get_gender_from_name(name, name_gender_df):
    name_row = name_gender_df[name_gender_df['Name'] == name]
    
    if name_row.empty:
        return None
    else:
        return name_row['Gender'].values[0]
    

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



def preprocess_movie_metadata(df):
    # here, we filter the movies on their date only
    df = to_datetime(df, column='movie_release_date')
    # filtering the years because we have very small data prior to 1910 and 2013
    df = df[(df['movie_release_date'] > 1913) & (df['movie_release_date'] <= 2013)]

    return df


def preprocess_char_metadata(df):

    # we convert the columns to datetime
    df = to_datetime(df, column='movie_release_date')

    df = df[(df['movie_release_date'] >= 1913) & (df['movie_release_date'] < 2013)]
    # drop duplicates
    df = df.drop_duplicates(subset=["wikipedia_movie_id", "char_name", "actor_name"]).reset_index(drop=True)
    # drop rows with missing values on both actor_name and actor_gender
    df = df.dropna(subset=["actor_name", "actor_gender"], how="all").reset_index(drop=True)
    # we fill the gender thanks to the actor_name and using an external dataset
    # we load the dataset
    name_gender = pd.read_csv("src/data/external_data/name_gender_dataset.csv")
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
        
    return df


# df1 = movies_metadata, df2 = character_metadata
def get_intersection(df1, df2): 
    """ Take raw dataframes, preprocess them and return the intersection 
    of the wikipedia_movie_id column of both dataframes."""
    
    # preprocess the dataframes
    df1 = preprocess_movie_metadata(df1)
    df2 = preprocess_char_metadata(df2)
    
    # get the intersection of wikipedia_movie_id
    # keep only the ids
    movies_id = df1[['wikipedia_movie_id']]
    # in character file, we must drop duplicates
    character_movie_id = df2.drop_duplicates(subset=['wikipedia_movie_id'])
    # keep only the ids
    character_movie_id = character_movie_id[['wikipedia_movie_id']]
    # merge the two dataframes to get the intersection of the wikipedia_movie_id column
    merged = pd.merge(movies_id, character_movie_id, how='inner', on='wikipedia_movie_id')
    # we keep only the rows that are in the intersection
    movies_metadata_intersection = df1['wikipedia_movie_id'].apply(lambda x : x in merged['wikipedia_movie_id'].values)
    df1 = df1[movies_metadata_intersection]
    # we keep only the rows that are in the intersection
    character_metadata_intersection = df2['wikipedia_movie_id'].apply(lambda x : x in merged['wikipedia_movie_id'].values)
    df2 = df2[character_metadata_intersection]
    # storing the intersection ids of wikipedia_movie_id
    ids = df1[["wikipedia_movie_id"]]
    
    return df1, df2, ids



def process_imdb_ratings(cmu_df):
    """ cmu_df is the dataframe containing the movies metadata """
    
    # Load the data
    # the title.basics.tsv and title.ratings.tsv files are too big to be stored in the repository
    imdb_title_df = pd.read_csv("src/data/external_data/raw_data/title.basics.tsv", sep='\t')
    imdb_ratings_df = pd.read_csv("src/data/external_data/raw_data/title.ratings.tsv", sep='\t')
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
    imdb_in_cmu.to_csv("data/imdb_ratings.csv", index=False)
    
    return imdb_in_cmu


def get_wikipedia_infobox_by_id(page_id):
    # Define the URL for accessing the Wikipedia page via the page ID
    url = f"https://en.wikipedia.org/w/api.php?action=parse&pageid={page_id}&format=json"
    headers = {
        "User-Agent": "MyWikipediaApp/1.0 (contact@example.com)"  # Update with your information
    }
    
    # Make a request to the Wikipedia API
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print("Error: Could not retrieve page.")
        return None

    # Parse the JSON response
    json_data = response.json()

    # Check if the "parse" key is in the response (indicating successful retrieval)
    if "parse" not in json_data:
        print("Error: Page data not found in response.")
        return None

    # Get the HTML content of the page
    html_content = json_data["parse"]["text"]["*"]

    # Use BeautifulSoup to parse the HTML and find the infobox
    soup = BeautifulSoup(html_content, 'html.parser')
    infobox = soup.find('table', {'class': 'infobox'})

    # Extract data from the infobox if it exists
    if infobox:
        infobox_data = {}
        
        # Go through each row in the infobox table
        for row in infobox.find_all('tr'):
            header = row.find('th')
            data = row.find('td')
            
            if header and data:
                header_text = header.get_text(" ", strip=True)
                data_text = data.get_text(" ", strip=True)
                infobox_data[header_text] = data_text
        return infobox_data
    else:
        print("No infobox found on the page.")
        return None
    
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
