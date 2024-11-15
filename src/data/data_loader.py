import pandas as pd

def load_movies_metadata(DATA_FOLDER_PATH):
    movies_metadata_df = pd.read_csv(DATA_FOLDER_PATH+'movie.metadata.tsv', sep='\t', header=None, 
                     names=['wikipedia_movie_id', 'freebase_movie_id', 'movie_name', 'movie_release_date',
                            'movie_box_office_revenue', 'movie_runtime', 'movie_languages', 'movie_countries', 'movie_genres'])
    return movies_metadata_df

def load_plot_summaries(DATA_FOLDER_PATH):
    data = []
    with open(DATA_FOLDER_PATH+'plot_summaries.txt', 'r', encoding='utf-8') as file:
        for line in file:
            row = line.strip().split('\t')
            data.append(row)
    plot_summaries_df = pd.DataFrame(data, columns=['wikipedia_movie_id', 'plot_summary'])
    return plot_summaries_df

def load_char_metadata(DATA_FOLDER_PATH):
    characters_metadata_df = pd.read_csv(DATA_FOLDER_PATH+'character.metadata.tsv', sep='\t', header=None, 
                 names=['wikipedia_movie_id', 'freebase_movie_id', 'movie_release_date', 'char_name',
                        'actor_date_of_birth', 'actor_gender', 'actor_height', 'actor_ethnicity', 'actor_name', 'actor_age', 
                        'char_actor_id', 'char_id', 'actor_id'])
    return characters_metadata_df


def load_tvtropes_data(DATA_FOLDER_PATH):
    tvtropes_data_df = pd.read_csv(DATA_FOLDER_PATH+'tvtropes.clusters.txt', sep='\t', header=None, 
                             names=['character_type', 'metadata'])
    return tvtropes_data_df
def load_name_clusters(DATA_FOLDER_PATH):
    names_df = pd.read_csv(DATA_FOLDER_PATH+'name.clusters.txt', sep='\t', header=None, 
                    names=['Cluster_name', 'Char_actor_id'])
    return names_df