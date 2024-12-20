import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# import folium
# import geopandas as gpd
# from folium import Choropleth, CircleMarker, Popup
# import pandas as pd
from plotly.subplots import make_subplots
# import ast
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# from scipy.stats import pearsonr, spearmanr

# CONSTANT DEFINITIONS
COLOR_MALE = '#2D9884'
COLOR_FEMALE = '#8059A4'
COLOR_MALE_LIGHT = '#17D07D'
COLOR_FEMALE_LIGHT = '#C48BF0'
COLOR_NEUTRAL = '#FDE047'
COLOR_PALETTE = {'M': COLOR_MALE, 'F': COLOR_FEMALE}
COLOR_PALETTE_LIST = [COLOR_FEMALE, COLOR_MALE]

COLOR_BECHDEL_PASSED = "#b7e1a1"
COLOR_BECHDEL_FAILED = "#f29e8e"

LABELS = {"M":"Male",
          "F":"Female",
          "number_of_movies":"Number of Movies", 
          "movie_genres":"Movie Genres",
          "movie_genre":"Movie Genre", 
          "country":"Country", 
          "char_name":"Number of Characters per movie",
          "actor_genders":"Actor Genders",
          "movie_release_date":"Release Year", 
          "movie_box_office":"Box Office", 
          "movie_runtime":"Runtime", 
          "movie_imdb_rating":"IMDB Rating", 
          "num_votes":"Number of Votes", 
          "average_rating":"Average Rating", 
          "director_gender":"Director Gender", 
          "box_office_revenue":"Movie Box Office Revenue", 
          "movie_budget":"Movie Budget",
          "movie_rendement":"Movie Rendement",
          'value': 'Proportion (%)', 
          'Emotion': 'Emotion'}

MALE = "M"
FEMALE = "F"

ONE = 1


# PLOTTING FUNCTIONS
# 2
def movies_by_country(df):
    """
    Plot the distribution of movies by country of production

    Args:
        df (DataFrame): Processed datafram on which we extract the data from
    """
    fig = px.bar(df, x="country", y="number_of_movies",
                 title='Distribution of Movies by Country', labels=LABELS)
    fig.update_traces(textfont_size=10, cliponaxis=False, marker_color=COLOR_NEUTRAL)
    fig.show()
    
# 2
def movies_by_genre(df):
    """
    Plot the distribution of movies by genre

    Args:
        df (DataFrame): Processed datafram on which we extract the data from
    """
    fig = px.bar(df, x="movie_genre", y="number_of_movies", 
                 title='Distribution of Movies by Genre', labels=LABELS)
    fig.update_traces(textfont_size=10, cliponaxis=False, marker_color=COLOR_NEUTRAL)
    fig.update_xaxes(tickangle=40)  # Set the angle of x-axis labels
    fig.show()
    
# 2
def movies_per_year(df):
    """
    Plot the distribution of movies by release year

    Args:
        df (DataFrame): Processed datafram on which we extract the data from
    """
    plt.figure(figsize=(12, 6))
    plt.hist(df["movie_release_date"], bins=range(df["movie_release_date"].min(), df["movie_release_date"].max() + ONE), color=COLOR_NEUTRAL, edgecolor='white') # add one to max to include it
    plt.xlabel('Release Year')
    plt.ylabel('Number of Movies')
    plt.title('Number of Movies Released by Year')
    plt.show()  
    
# 3.A & 3.B 1)
def plot_gender_distribution(df, gender_column):
    """
    Plot the gender distribution of movie directors / characters (depending on the "gender_column" parameter)

    Args:
        df (DataFrame): Processed dataframe on which we extract the data from
        gender_column (str): column which data is to be extracted from
    """
    # Exploding the gender column and calculating the counts
    gender_counts = df[gender_column].explode().value_counts()

    # Calculating percentages
    total_characters = gender_counts.sum()
    percentages = (gender_counts / total_characters) * 100
    
    label = "Movie Characters" if gender_column == "actor_genders" else LABELS.get(gender_column) 

    # Create the bar chart with Plotly
    fig = px.bar(
        x=gender_counts.index,  # Gender categories
        y=gender_counts.values,  # Counts
        labels={'x': label, 'y': 'Number'},
        title=f"Gender Distribution of {label}",
        color=gender_counts.index,  # Color by gender
        color_discrete_map=COLOR_PALETTE 
    )
    
    fig.update_layout(legend_title_text="Director Gender")
    
    
    
    fig.update_traces(hovertemplate=(
            "Gender: %{x}<br>"
            "Number : %{y}<br>" 
    ))
    
    y_annotation = 750 if gender_column == "director_gender" else 7000

    # Add percentage text on top of bars
    for index, (count, percentage) in enumerate(zip(gender_counts.values, percentages)):
        fig.add_annotation(
            x=index,
        y=count + y_annotation,  # Shift the annotation upwards
            text=f'{percentage:.1f}%', 
            showarrow=False, 
            font=dict(size=12), 
            align='center', 
            valign='bottom'
        )

    # Show the plot
    fig.show()
    
# 3.B 2)
def age_actors_by_dir(fig, data, title, color, dash):
    """
    Plot the average age of actors in movies directed by a movie director of a specific gender

    Args:
        fig (Figure): Graph Object Figure 
        data (DataFrame): Processed dataframe on which we extract the data from
        title (str): title of the plot
        color (str): color of the plot
        dash (str): indicated whether the line is dashed or not
    """
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data.values,
        mode='lines',
        name=title,
        line=dict(color=color, dash=dash),
        
        hovertemplate=(
            "Age: %{x}<br>"
            "Percentage of actors : %{y:.2f}%<br><br>"
        )
    ))

# 2.B 2)  
def age_actors_layout(fig):
    """
    Update the layout of the age actors distribution plot

    Args:
        fig (Figure): Graph Object Figure 
    """
    fig.update_layout(
        title="Age Distribution of Actors by Director Gender (Percentage)",
        xaxis_title="Age",
        yaxis_title='Actors [%]',
        legend_title="Actor Gender & Director Gender",
        template="plotly_white"
    )
    fig.show()
    
# 3.B 3)    
def fem_representation_by_dir(df):   
    """
    Plot the percentage of female characters in movies directed by male and female directors over time,
    including the number of movies, characters, and directors.

    Args:
    df (pd.DataFrame): Processed dataframe on which we extract the data from
    """
    # Filter movies from 1930 onwards
    df = df[df["movie_release_date"] >= 1930]

    # Split the DataFrame by director gender
    male_directors = df[df["director_gender"] == MALE]
    female_directors = df[df["director_gender"] == FEMALE]

    # Compute percentage of female characters per year
    char_female_male = (male_directors.groupby("movie_release_date")["char_F"].sum().values / 
                        male_directors.groupby("movie_release_date")["char_tot"].sum().values) * 100
    char_female_female = (female_directors.groupby("movie_release_date")["char_F"].sum().values / 
                          female_directors.groupby("movie_release_date")["char_tot"].sum().values) * 100

    # Total number of movies per year
    movies_count_male = male_directors.groupby("movie_release_date")["wikipedia_movie_id"].nunique().values
    movies_count_female = female_directors.groupby("movie_release_date")["wikipedia_movie_id"].nunique().values

    # Number of directors per year
    directors_count_male = male_directors.groupby("movie_release_date")["director_gender"].count().values
    directors_count_female = female_directors.groupby("movie_release_date")["director_gender"].count().values

    # Extract years (x-axis values)
    years = np.array(sorted(male_directors["movie_release_date"].unique()))

    # Create the Plotly figure
    fig = go.Figure()

    # Male-directed movies
    fig.add_trace(go.Scatter(
        x=years,
        y=char_female_male,
        name="Male Directors",
        mode="lines",
        marker=dict(color=COLOR_MALE),
        customdata=np.stack((male_directors.groupby("movie_release_date")["char_tot"].sum().values, 
                             movies_count_male, directors_count_male), axis=-1),
        hovertemplate=(
            "Year: %{x}<br>"
            "Proportion of Female Characters: %{y:.2f}%<br><br>"
            "Total number of Characters: %{customdata[0]}<br>"
            "Number of Movies: %{customdata[1]}<br>"
            "Number of Directors: %{customdata[2]}"
        ),
    ))

    # Female-directed movies
    fig.add_trace(go.Scatter(
        x=years,
        y=char_female_female,
        name="Female Directors",
        mode="lines",
        marker_color=COLOR_FEMALE,
        customdata=np.stack((female_directors.groupby("movie_release_date")["char_tot"].sum().values, 
                             movies_count_female, directors_count_female), axis=-1),
        hovertemplate=(
            "Year: %{x}<br>"
            "Proportion of Female Characters: %{y:.2f}%<br><br>"
            "Number of Characters: %{customdata[0]}<br>"
            "Number of Movies: %{customdata[1]}<br>"
            "Number of Directors: %{customdata[2]}"
        ),
    ))

    # Customize layout
    fig.update_layout(
        title="Representation of Female Characters by Director Gender",
        yaxis_title="Female Characters in Movies [%]",
        xaxis_title="Year",
        xaxis=dict(
            tickmode='array',
            tickvals=years[::5],  # Display ticks every 5 years
            ticktext=years[::5]
        ),
        yaxis=dict(
            tickformat=".0f",  # Format y-axis as whole percentages
        ),
        legend_title="Director Gender",
        width=1150,
        height=700,
        template="plotly_white"  # White background for clarity
    )

    # Show the plot
    fig.show()
  
# 3.B 4)  
def map_fem_char(gender_percentages_per_country, director_gender):
    """
    Generates a choropleth map of average female character percentages, 
    total characters, and movie counts in movies directed by directors of the specified gender, grouped by country.

    Parameters:
    - movies_df: DataFrame containing movie data.
    - director_gender: Gender of the directors to filter ('M' or 'F').

    Returns:
    - Folium Map object.
    """
    # Load the world map
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

    # Map country names to align with GeoPandas world map
    country_mapping = {
        "Malta": "Malta",
        "United States": "United States of America",
        "Korea": "South Korea",
        "Republic of Macedonia": "North Macedonia",
        "Bosnia and Herzegovina": "Bosnia and Herz.",
        "Crime": "Crimea",
        "Singapore": "Singapore",
        "Burma": "Myanmar",
        "Czech Republic": "Czechia",
    }

    # Replace country names in gender_percentages_per_country
    gender_percentages_per_country["movie_countries"] = gender_percentages_per_country["movie_countries"].replace(country_mapping)

    # Merge gender data with GeoPandas world map
    world = world.merge(gender_percentages_per_country, left_on="name", right_on="movie_countries", how="left")

    # Initialize the Folium map
    m = folium.Map(
        location=[0, 0],
        zoom_start=2,
        tiles="cartodbpositron",
        no_wrap=True,
        continuous_world=False
    )

    # Add a choropleth layer
    Choropleth(
        geo_data=world,
        data=gender_percentages_per_country,
        columns=["movie_countries", "mean_female_percentage"],
        key_on="feature.properties.name",
        fill_color="RdYlGn",
        fill_opacity=0.7,
        line_opacity=0.2,
        nan_fill_color="lightgray",
        legend_name=f"Mean Female Characters in Movies Directed by {director_gender} (%)",
    ).add_to(m)

    # Adjusted coordinates for certain countries (optional)
    adjusted_coordinates = {
        "France": [46.603354, 1.888334],
        "Russia": [61.52401, 105.318756],
        "United States": [37.09024, -95.712891],
        "Japan": [36.204824, 138.252924],
        "Malaysia": [4.210484, 101.975766],
        "Indonesia": [-2.548926, 118.014863],
    }

    # Add tooltips for additional data
    for _, row in world.iterrows():
        if pd.notna(row["mean_female_percentage"]):  # Ensure valid data
            # Use adjusted coordinates if available, otherwise use the centroid
            location = adjusted_coordinates.get(
                row["name"],
                [row.geometry.centroid.y, row.geometry.centroid.x]
            )
            CircleMarker(
                location=location,
                radius=3,
                color="skyblue",
                fill=True,
                fill_opacity=0.8,
                popup=Popup(
                    f"""
                    <b>{row['name']}</b><br>
                    Mean Female Percentage: {row['mean_female_percentage']:.1f}%<br>
                    Total Movies: {int(row.get('movies_count', 0)) if pd.notna(row.get('movies_count')) else 'N/A'}<br>
                    Total Characters: {int(row.get('total_characters', 0)) if pd.notna(row.get('total_characters')) else 'N/A'}
                    """,
                    max_width=300
                )
            ).add_to(m)
            
    return m

# 3.B 5)
def plot_top10_genres(df, gender_real):
    """
    Plot the gender representation across the 10 Movie Genres most represented

    Args:
        df (DataFrame): Processed dataframe on which we extract the data from
        gender_real (str): Gender of the movie director
    """

    
    fig = go.Figure()
    # male characters
    fig.add_trace(go.Bar(
        x=df.index,
        y=df[FEMALE],
        name=FEMALE,
        marker_color=COLOR_FEMALE,
        
        hovertemplate=(
            "Movie Genre: %{x}<br>"
            "Proportion of Female Characters: %{y:.2f}%<br>"
        )
    ))
    
    # female characters
    fig.add_trace(go.Bar(
        x=df.index,
        y=df[MALE],
        name=MALE,
        marker_color=COLOR_MALE,
        
        hovertemplate=(
            "Movie Genre: %{x}<br>"
            "Proportion of Male Characters: %{y:.2f}%<br><br>"
        )
    ))

    fig.update_layout(
        title=f"Gender Representation Across Top-10 Movie Genres - {gender_real} Director",
        xaxis_title="Movie Genre",
        yaxis_title="Percentage of Characters (%)",
        barmode='stack',
        xaxis=dict(
            tickmode='array',
            tickvals=df.index,
            ticktext=df.index,
            tickangle=45
        ),
        height=600,
        legend_title=" Character Gender",
        legend=dict(title="Character Gender", orientation="h", x=0.5, xanchor="center", y=1.1)
    )
    fig.show()
    
# 3.B 5)
def genres_most_fem_char(df, director_gender, sort, title):
    """
    Plots the top 5 genres with the highest/lowest percentage of female characters depending on the gender of the movie director

    Args:
        df (DataFrame): Processed dataframe on which we extract the data from
        director_gender (str): "Male" or "Female"
        sort (bool): True for ascending, False for descending order (to sort percentage of feminine representation)
        title (str): to adapt the title of the graph
    """
    # filter movies within two dates
    movies_df = df[(df["movie_release_date"] >= 1990) & (df["movie_release_date"] <= 2010)]
    movies_df = movies_df.explode("movie_genres")
    movies_df["female_percentage"] = (movies_df["char_F"] / movies_df["char_tot"]) * 100
    mean_female_percentage_per_genre = movies_df.groupby("movie_genres")["female_percentage"].mean()
    number_movies_per_genre = movies_df.groupby("movie_genres")["wikipedia_movie_id"].nunique()
    total_characters_per_genres = movies_df.groupby("movie_genres")["char_tot"].sum()
    
    # Filter to include only genres with at least 100 movies
    top_genres_number_movies = number_movies_per_genre[number_movies_per_genre.values > 100].index
    top_genres = mean_female_percentage_per_genre[top_genres_number_movies].sort_values(ascending=sort).head(10).index
    female_percentage_top = mean_female_percentage_per_genre[top_genres]
    
    # Filter for top genres
    number_movies_top = number_movies_per_genre[top_genres]
    tot_char_top = total_characters_per_genres[top_genres]
    
    # Plot the data
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=female_percentage_top.index,
        y=female_percentage_top,
        name="Female",
        marker_color=COLOR_PALETTE.get(director_gender[0]), # grab first letter that defines gender
        textposition='auto'  # Display the text directly on the bars
    ))
    
    # Update the layout
    fig.update_layout(
        title=f"Top-5 Genres with {title} Percentage of Female Characters - {director_gender} Director",
        xaxis_title="Movie Genre",
        yaxis_title="Percentage of Female Characters (%)",
        xaxis=dict(
            tickmode='array',
            tickvals=female_percentage_top.index,
            ticktext=female_percentage_top.index,
            tickangle=45
        ),
        height=600
    )
    
    # Add hover details
    fig.update_traces(
        customdata=np.stack((tot_char_top.values,
                            number_movies_top.values, 
                        ), axis=-1),
        hovertemplate=(
            "Genre: %{x}<br>"
            "Mean proportion of Female Characters: %{y:.2f}%<br>"
            "Number of Characters: %{customdata[0]}<br>"
            "Number of Movies: %{customdata[1]}<br>"
            # "Percentage of Movies in Genre: %{customdata[2]:.2f}%"
        )
    )
    
    fig.show()
    
# 3.C
def plot_director_trope_pie_charts(male_director_data, female_director_data):
    """
    Plot the gender distribution of TV Tropes representation by director gender

    Args:
        male_director_data (DataFrame): Processed dataframe on which we extract the data from, specifically male director data
        female_director_data (DataFrame): Processed dataframe on which we extract the data from, specifically female director data
    """
    fig = go.Figure()
    # Male Tropes Pie Chart
    fig.add_trace(go.Pie(
        labels=male_director_data.index,
        values=male_director_data.values,
        hole=0.5,  # Adds a donut hole for aesthetics
        domain=dict(x=[0, 0.5]),  # Place pie chart on the left
        textinfo='none',
        marker=dict(colors=COLOR_PALETTE_LIST),        
        name="Male directors",
        hovertemplate='%{label}: %{value:.2f}%'  # Customize hovertemplate (show label and percentage with 2 decimals)
    ))

    # Female Tropes Pie Chart
    fig.add_trace(go.Pie(
        labels=female_director_data.index,  # 'Male' and 'Female' tropes
        values=female_director_data.values,
        hole=0.5,
        domain=dict(x=[0.5, 1]),  # Place pie chart on the right
        textinfo='none',
        marker=dict(colors=COLOR_PALETTE_LIST),     
        name="Female directors",
        hovertemplate='%{label}: %{value:.2f}%'  # Customize hovertemplate (show label and percentage with 2 decimals)
    ))

    fig.update_layout(
        title="Representation of Male/Female Tropes by Director Gender",
        grid=dict(rows=1, columns=2),
        annotations=[
            dict(text='Male Directors', x=0.25, y=1.05, font_size=14, showarrow=False, xanchor="center", yanchor="bottom"),
            dict(text='Female Directors', x=0.75, y=1.05, font_size=14, showarrow=False, xanchor="center", yanchor="bottom")
        ],
        showlegend=True,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.2),
    )

    fig.show()
   
# 3.C
def gender_char_types(df):
    """
    Plot the gender distribution of actors of different tv tropes regarding director gender of the movie

    Args:
        df (DataFrame): Processed dataframe on which we extract the data from
    """
    # Set up the number of subplots
    categories_per_subplot = 12
    num_subplots = (len(df) + categories_per_subplot - 1) // categories_per_subplot

    fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 4 * num_subplots))

    # Plot each subset in a separate subplot
    for i in range(num_subplots):
        start = i * categories_per_subplot
        end = start + categories_per_subplot
        subset = df.iloc[start:end]

        subset.plot(kind='bar', stacked=True, ax=axes[i], color=[COLOR_PALETTE])
        axes[i].set_title(f"Repartition of Genders for Character Types")
        axes[i].set_xlabel("")  # Character type
        axes[i].set_ylabel("Number of Actors")
        axes[i].legend(title="Gender")
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.show()
 
# 3.D 1)
import plotly.graph_objects as go

def bechdel_4_categories(df):
    """
    Plot the results of the Bechdel test ratings regarding the gender of the director,
    using labels for 'Bechdel Failed', 'Bechdel Passed', 'Bechdel Partially Passed', and 'Bechdel Undetermined'.

    Args:
        df (DataFrame): Processed dataframe on which we extract the data from
    """
    # Mapping for Bechdel ratings
    bechdel_mapping = {0: "No 2 women", 
                       1: "2 women not talking", 
                       2: "2 women talking about a man", 
                       3: "2 women talking about something else than a man"}
    
    # Calculate histograms for male and female directors
    male_director_hist = (
        df[df['director_gender'] == "M"]['bechdel_rating']
        .replace(bechdel_mapping)  # Replace values with labels
        .value_counts(normalize=True) * 100
    )
    female_director_hist = (
        df[df['director_gender'] == "F"]['bechdel_rating']
        .replace(bechdel_mapping)
        .value_counts(normalize=True) * 100
    )
    
    # Labels for x-axis (based on the bechdel_mapping)
    x = list(bechdel_mapping.values())
    
    # Create bar traces for Male and Female Directors
    trace_male = go.Bar(
        x=x,
        y=[male_director_hist.get(label, 0) for label in x],
        name='Male Directors',
        marker_color=COLOR_MALE,
        width=0.4,
        customdata=['Male Directors'] * len(x),
        hovertemplate=(
        "%{x}<br>"
        "%{customdata}<br>"
        "Proportion of movies: %{y:.2f}%<br>" 
        )
    )
    trace_female = go.Bar(
        x=x,
        y=[female_director_hist.get(label, 0) for label in x],
        name='Female Directors',
        marker_color=COLOR_FEMALE,
        width=0.4,
        customdata=['Female Directors'] * len(x),
        hovertemplate=(
        "%{x}<br>"
        "%{customdata}<br>"
        "Proportion of movies: %{y:.2f}%<br>" 
        )
    )
    
    # Create layout
    layout = go.Layout(
        title="Bechdel Test Ratings by Gender of Directors",
        xaxis=dict(title="Bechdel Rating", tickvals=x, ticktext=list(bechdel_mapping.keys())),
        yaxis=dict(title="Movies [%]"),
        barmode='group',  # Group the bars
        legend_title="Director Gender",
        bargap=0.2,  # Gap between bars
        template="plotly_white"
    )
    
    # Create figure and show plot
    fig = go.Figure(data=[trace_male, trace_female], layout=layout)
    fig.show()

def bechdel_2_categories(df):
    """
    Plot the results of the Bechdel test ratings regarding the gender of the director,
    using labels for 'Bechdel Failed' and 'Bechdel Passed'.

    Args:
        df (DataFrame): Processed dataframe on which we extract the data from
    """
    # Mapping for Bechdel ratings
    bechdel_mapping = {0: "Bechdel Failed", 1: "Bechdel Passed"}
    
    # Calculate histograms for male and female directors
    male_director_hist = (
        df[df['director_gender'] == 0]['bechdel_rating']
        .replace(bechdel_mapping)  # Replace values with labels
        .value_counts(normalize=True) * 100
    )
    female_director_hist = (
        df[df['director_gender'] == 1]['bechdel_rating']
        .replace(bechdel_mapping)
        .value_counts(normalize=True) * 100
    )
    
    # Labels for x-axis
    x = list(bechdel_mapping.values())
    
    # Create bar traces for Male and Female Directors
    trace_male = go.Bar(
        x=x,
        y=[male_director_hist.get(label, 0) for label in x],
        name='Male',
        marker_color=COLOR_MALE,
        width=0.4,
        customdata=['Male Directors'] * len(x),
        hovertemplate=(
            "%{x}<br>"
            "%{customdata}<br>"
            "Proportion of movies : %{y:.2f}%<br>" 
    )
    )
    trace_female = go.Bar(
        x=x,
        y=[female_director_hist.get(label, 0) for label in x],
        name='Female',
        marker_color=COLOR_FEMALE,
        width=0.4,
        customdata=['Female Directors'] * len(x),
        hovertemplate=(
            "%{x}<br>"
            "%{customdata}<br>"
            "Proportion of movies : %{y:.2f}%<br>" 
    )
    )
    
    # Create layout
    layout = go.Layout(
        title="Bechdel Test Ratings by Gender of Directors",
        xaxis=dict(title="Bechdel Rating", tickvals=x),
        yaxis=dict(title="Movies [%]"),
        barmode='group',  # Group the bars
        legend_title="Director Gender",
        bargap=0.2  # Gap between bars
    )
    
    # Create figure and show plot
    fig = go.Figure(data=[trace_male, trace_female], layout=layout)
    fig.show()

    
# 3.D 2)
def graph_emotions(emotion_totals):
    """
    Plot the emotions distribution in the plot summaries.

    Args: 
        emotion_totals (dict): Processed dictionary on which we extract the data from

    """
    df = pd.DataFrame(list(emotion_totals.items()), columns=['Emotion', 'Score'])

    fig = px.pie(df, names='Emotion', values='Score', title='Average emotions distribution in plot summaries',
                 color_discrete_sequence=px.colors.qualitative.Set2,
                 hover_data={'Emotion': True, 'Score': True}
            )
    
    fig.update_traces(hovertemplate=
                    "Emotion: %{label}<br>"
                    "Score: %{value:.2f}<br>" )
    
    fig.update_layout(legend_title_text="Emotion")
    
    fig.show()

# 3.D 2)
def plot_bechdel_predictions(df_bechdel_predictions):
    """
    Plot the distribution of the bechdel predictions

    Args: 
        df_bechdel_predictions (DataFrame): Dataframe with the bechdel predictions
    Returns:
        None
    """
    value_counts = df_bechdel_predictions['bechdel_rating'].value_counts().reset_index()
    value_counts.columns = ['bechdel_rating', 'count']

    # Apply the bechdel_mapping to replace the values
    bechdel_mapping = {0: "Bechdel Failed", 1: "Bechdel Passed"}
    value_counts['bechdel_rating'] = value_counts['bechdel_rating'].map(bechdel_mapping)

    # Plotting
    fig = px.bar(
        value_counts,
        x='bechdel_rating',
        y='count',
        text='count',
        labels={'bechdel_rating': 'Bechdel Result', 'count': 'Number of movies'},
        title='Predictions - Distribution of Bechdel Result',
        color_discrete_sequence=[COLOR_NEUTRAL],
        
        hovertemplate=(
            "Bechdel Result: %{x}<br>"
            "Number of movies: %{y}<br>"
        )
    )

    fig.update_traces(textposition='outside')
    fig.show()


def bechdel_test_obtain_with_ml_ratings_by_gender(df):
    """
    Plot the results of the Bechdel test ratings regarding the gender of the director,
    using labels for 'Bechdel Failed' and 'Bechdel Passed'.

    Args:
        df (DataFrame): Processed dataframe on which we extract the data from
    """
    # Mapping for Bechdel ratings
    bechdel_mapping = {0: "Bechdel Failed", 1: "Bechdel Passed"}
    
    # Calculate histograms for male and female directors
    male_director_hist = (
        df[df['director_gender'] == 0]['bechdel_rating']
        .replace(bechdel_mapping)  # Replace values with labels
        .value_counts(normalize=True) * 100
    )
    female_director_hist = (
        df[df['director_gender'] == 1]['bechdel_rating']
        .replace(bechdel_mapping)
        .value_counts(normalize=True) * 100
    )
    
    # Labels for x-axis
    x = list(bechdel_mapping.values())
    
    # Create bar traces for Male and Female Directors
    trace_male = go.Bar(
        x=x,
        y=[male_director_hist.get(label, 0) for label in x],
        name='Male Directors',
        marker_color=COLOR_MALE,
        width=0.4,
        
        hover_template=(
            "{x}<br>"
            "Proportion of movies : {y:.2f}%<br>"
        )
    )
    trace_female = go.Bar(
        x=x,
        y=[female_director_hist.get(label, 0) for label in x],
        name='Female Directors',
        marker_color=COLOR_FEMALE,
        width=0.4,
        
        hover_template=(
            "{x}<br>"
            "Proportion of movies : {y:.2f}%<br>"
        )
    )
    
    # Create layout
    layout = go.Layout(
        title="Predictions - Bechdel Result by Gender of Directors",
        xaxis=dict(title="Bechdel Result"),
        yaxis=dict(title="Movies [%]"),
        barmode='group',  # Group the bars
        legend_title="Director Gender",
        bargap=0.2  # Gap between bars
    )
    
    # Create figure and show plot
    fig = go.Figure(data=[trace_male, trace_female], layout=layout)
    fig.show()
    
# 3.D 2)
def graph_emotions_bechdel_combined(df_bechdel):
    """
    Plot the emotions distribution regarding the results of the bechdel test

    Args:
        df_bechdel (DataFrame): Processed dataframe on which we extract the data from
        
    """
    # Calculation of the average emotions for each DataFrame
    def compute_mean_emotions(df):
        emotions = ["neutral", "sadness", "anger", "fear", "disgust", "surprise", "joy"]
        return df[emotions].mean()

    # creation of datasets that pass or fail the bechdel test
    bechdel_grade3 = df_bechdel[df_bechdel["bechdel_rating"]==1]
    bechdel_grade012 = df_bechdel[df_bechdel["bechdel_rating"]!=1]

    # Creation of datasets by director gender
    bechdel_grade3_men = bechdel_grade3[bechdel_grade3["director_gender"]==0]
    bechdel_grade3_women = bechdel_grade3[bechdel_grade3["director_gender"]==1]
    bechdel_grade012_men = bechdel_grade012[bechdel_grade012["director_gender"]==0]
    bechdel_grade012_women = bechdel_grade012[bechdel_grade012["director_gender"]==1]

    # Data for the graph
    data_women_grade3 = compute_mean_emotions(bechdel_grade3_women)
    data_men_grade3 = compute_mean_emotions(bechdel_grade3_men)

    data_women_grade012 = compute_mean_emotions(bechdel_grade012_women)
    data_men_grade012 = compute_mean_emotions(bechdel_grade012_men)

    # Creation of a graph with two subplots side by side
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'polar'}, {'type': 'polar'}]],      
    )
    

    # First graph: For Male directors
    fig.add_trace(go.Scatterpolar(
        r=data_men_grade3,
        theta=data_men_grade3.index,
        fill='toself',
        name='Passed',
        mode='markers',
        marker_color=COLOR_BECHDEL_PASSED,
        
        hovertemplate=(
            "Emotion: %{theta}<br>"
            "Proportion: %{r:.2f}%<br>"
    )
    ), row=1, col=2)

    fig.add_trace(go.Scatterpolar(
        r=data_men_grade012,
        theta=data_men_grade012.index,
        fill='toself',
        name='Failed',
        marker_color=COLOR_BECHDEL_FAILED,
        
        hovertemplate=(
            "Emotion: %{theta}<br>"
            "Proportion: %{r:.2f}%<br>"
    )
    ), row=1, col=2)

    # Second graph: For Female directors
    fig.add_trace(go.Scatterpolar(
        r=data_women_grade3,
        theta=data_women_grade3.index,
        fill='toself',
        name='Passed',
        marker_color=COLOR_BECHDEL_PASSED,
        
        hovertemplate=(
            "Emotion: %{theta}<br>"
            "Proportion: %{r:.2f}%<br>"
    )
    ), row=1, col=1)

    fig.add_trace(go.Scatterpolar(
        r=data_women_grade012,
        theta=data_women_grade012.index,
        fill='toself',
        name='Failed',
        marker_color=COLOR_BECHDEL_FAILED,
        
        hovertemplate=(
            "Emotion: %{theta}<br>"
            "Proportion: %{r:.2f}%<br>"
    )
    ), row=1, col=1)

    fig.update_layout(
        title_text="Emotion Distribution by Gender for Bechdel Test Results",
        showlegend=True,
        polar=dict(
        radialaxis=dict(visible=True, range=[0, 0.24]),
        # Adjust the size of the radar plot area
        angularaxis=dict(tickmode='array', tickvals=data_men_grade3.index)  # Adjust tick values for readability
    ),
        template="plotly_white",
        annotations=[
            dict(text="Female Director", x=0.22, y=1.09, showarrow=False, font=dict(size=14, color="black"), align="center"),
            dict(text="Male Director", x=0.86, y=1.09, showarrow=False, font=dict(size=14, color="black"), align="center")
        ]
    )


    fig.show()
    
# 3.D 2)
def graph_ratio_emotion_by_director_gender(ratios_women, ratios_men):
    """
    Plot the emotions ratio regarding the director gender in a histogram

    Args:
        ratios_women (DataFrame): percentage of emotions among female directors
        ratios_men (DataFrame): percentage of emotions among male directors
    """
    # Transformation into a DataFrame for Plotly
    df = pd.DataFrame({
        'Emotion': list(ratios_women.keys()),
        'Women': list(ratios_women.values()),
        'Men': list(ratios_men.values())
    })

    fig = px.bar(
        df,
        x='Emotion',
        y=['Women', 'Men'],
        title='Emotions distribution by director gender',
        labels=LABELS,
        barmode='group',
        color_discrete_map={
            "Women": COLOR_FEMALE,
            "Men": COLOR_MALE
        }
    )

    
    fig.update_traces(marker=dict(opacity=0.8),
                      hovertemplate=(
                          "Emotion: %{x}<br>"
                          "Proportion: %{y:.2f}%<br>"))
    fig.update_layout(
        xaxis_title="Emotion",
        yaxis_title="Emotion proportion [%]",
        legend_title="Director Gender",
        template="plotly_white"
    )

    fig.show()

# 3.D 2)
def graph_ratio_emotion_radar_by_director_gender(ratios_women, ratios_men):
    """
    Plot the emotions ratio regarding the director gender in a radar graph 

    Args:
        ratios_women (DataFrame): percentage of emotions among female directors
        ratios_men (DataFrame): percentage of emotions among male directors
    """
    emotions = list(ratios_women.keys())
    values_women = list(ratios_women.values())
    values_men = list(ratios_men.values())
    
    # To close the radar, add the first dot at the end
    emotions += [emotions[0]]
    values_women += [values_women[0]]
    values_men += [values_men[0]]
    

    fig = go.Figure()
    # Female directors
    fig.add_trace(go.Scatterpolar(
        r=values_women,
        theta=emotions,
        fill='toself',
        name='Female Directors',
        line=dict(color=COLOR_FEMALE),
        marker=dict(size=6),
        
        hovertemplate=(
            "Emotion: %{theta}<br>"
            "Proportion: %{r:.2f}%<br>"
    )
    ))
    # Male directors
    fig.add_trace(go.Scatterpolar(
        r=values_men,
        theta=emotions,
        fill='toself',
        name='Male Directors',
        line=dict(color=COLOR_MALE),
        marker=dict(size=6),
        
        hovertemplate=(
            "Emotion: %{theta}<br>"
            "Proportion: %{r:.2f}%<br>"
    )
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True),
        ),
        title='Ratio of Emotion in Plot Summaries by Director Gender',
        legend_title="Director Gender",
        template="plotly_white"
    )

    fig.show()
    
# 3.D 3)
def plot_confusion_matrix(y_test, y_pred_test):
    """
    Plot the confusion matrix of the Random Forest model

    Args:
        y_test : values of test set
        y_pred_test : values of predicted test set
    """
    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)

    # Normalize the confusion matrix to get proportions
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plotting the normalized confusion matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
    plt.title('Normalized Random Forest Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# 3.E 1)  
def avg_rating(df):
    """
    Plot the average rating of movies by gender of the movie director

    Args:
        df (DataFrame): Processed dataframe on which we extract the data from
    """
    fig = px.histogram(df, x="average_rating", color="director_gender", 
                 title="Proportional Average Rating of Movies by Director Gender", histnorm='percent',
                 barmode='group', nbins=40, color_discrete_map=COLOR_PALETTE,
                 labels=LABELS)
    fig.update_yaxes(title_text='Proportion of movies')
    fig.show()
 
# 3.E 1)    
def avg_rating_bechdel(ratings_passed_female, ratings_failed_female, ratings_passed_male, ratings_failed_male):
    """
    Plot the average rating of movies categorized as Optimal/Worst by gender of the movie director

    Args:
        df (DataFrame): Processed dataframe on which we extract the data from
    """
    # fig = go.Figure()

    # # Female Directors - Passed Bechdel Test
    
    # # to display when hovering
    # nb_movies_female_passed = len(ratings_passed_female)
    
    # fig.add_trace(go.Histogram(
    #         x=ratings_passed_female.values,
    #         xbins=dict(start=0, end=10, size=1),
    #         marker_color=COLOR_FEMALE,
    #         opacity=0.75,
    #         name="Bechdel Passed - Female Director",
    #         histnorm='percent',
            
    #         hovertemplate=(
    #             "Bechdel Passed - Female Director<br>"
    #             "Proportion: %{y:.2f}%<br>"
    #             f"Number of movies: {nb_movies_female_passed}<br>"
    # )
    #     )
    # )

    # # Female Directors - Failed Bechdel Test
    
    # # to display when hovering
    # nb_movies_female_failed = len(ratings_failed_female)
    
    # fig.add_trace(go.Histogram(
    #         x=ratings_failed_female.values,
    #         xbins=dict(start=0, end=10, size=1), 
    #         marker_color=COLOR_FEMALE_LIGHT,
    #         opacity=0.75,
    #         name="Bechdel Failed - Female Director",
    #         histnorm='percent',
            
    #          hovertemplate=(
    #             "Bechdel Passed - Male Director<br>"
    #             "Proportion: %{y:.2f}%<br>"
    #             f"Number of movies: {nb_movies_female_failed}<br>"
    # )
    #     )
    # )
    
    # # Male Directors - Passed Bechdel Test
    
    # # to display when hovering
    # nb_movies_male_passed = len(ratings_passed_male)
    
    # fig.add_trace(go.Histogram(
    #         x=ratings_passed_male.values,
    #         xbins=dict(start=0, end=10, size=1),
    #         marker_color=COLOR_MALE,
    #         opacity=0.75,
    #         name="Bechdel Passed - Male Director",
    #         histnorm='percent',
            
    #         hovertemplate=(
    #             "Bechdel Passed - Male Director<br>"
    #             "Proportion: %{y:.2f}%<br>"
    #             f"Number of movies: {nb_movies_male_passed}<br>"
    # )
    #     )
    # )

    # # Male Directors - Failed Bechdel Test
    
    # # to display when hovering
    # nb_movies_male_failed = len(ratings_failed_male)
    
    # fig.add_trace(go.Histogram(
    #         x=ratings_failed_male.values,
    #         xbins=dict(start=0, end=10, size=1),
    #         marker_color=COLOR_MALE_LIGHT,
    #         opacity=0.75,
    #         name="Bechdel Failed - Male Director",
    #         histnorm='percent',
            
    #         hovertemplate=(
    #             "Bechdel Passed - Male Director<br>"
    #             "Proportion: %{y:.2f}%<br>"
    #             f"Number of movies: {nb_movies_male_failed}<br>"
    # )
    #     )
    # )

    # # Personnalisation du graphique
    # fig.update_layout(
    #     title="Average Rating depending on Bechdel Test by Director Gender",
    #     xaxis_title="Average Rating",
    #     yaxis_title="Movies [%]"
    # )
    
    # fig.show()


    # Create a subplot grid with 1 row and 2 columns
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Female Directors", "Male Directors"],
        shared_yaxes=True  # This allows the y-axes to be shared between the two subplots
    )

    # Female Directors - Passed Bechdel Test
    nb_movies_female_passed = len(ratings_passed_female)
    fig.add_trace(go.Histogram(
            x=ratings_passed_female.values,
            xbins=dict(start=0, end=10, size=1),
            marker_color=COLOR_BECHDEL_PASSED,
            opacity=0.75,
            name="Bechdel Passed",
            histnorm='percent',
            hovertemplate=(
                "Bechdel Passed - Female Director<br>"
                "Proportion: %{y:.2f}%<br>"
                f"Number of movies: {nb_movies_female_passed}<br>"
            )
        ), row=1, col=1
    )

    # Female Directors - Failed Bechdel Test
    nb_movies_female_failed = len(ratings_failed_female)
    fig.add_trace(go.Histogram(
            x=ratings_failed_female.values,
            xbins=dict(start=0, end=10, size=1),
            marker_color=COLOR_BECHDEL_FAILED,
            opacity=0.75,
            name="Bechdel Failed",
            histnorm='percent',
            hovertemplate=(
                "Bechdel Failed - Female Director<br>"
                "Proportion: %{y:.2f}%<br>"
                f"Number of movies: {nb_movies_female_failed}<br>"
            )
        ), row=1, col=1
    )

    # Male Directors - Passed Bechdel Test
    nb_movies_male_passed = len(ratings_passed_male)
    fig.add_trace(go.Histogram(
            x=ratings_passed_male.values,
            xbins=dict(start=0, end=10, size=1),
            marker_color=COLOR_BECHDEL_PASSED,
            opacity=0.75,
            name="Bechdel Passed",
            histnorm='percent',
            hovertemplate=(
                "Bechdel Passed - Male Director<br>"
                "Proportion: %{y:.2f}%<br>"
                f"Number of movies: {nb_movies_male_passed}<br>"
            )
        ), row=1, col=2
    )

    # Male Directors - Failed Bechdel Test
    nb_movies_male_failed = len(ratings_failed_male)
    fig.add_trace(go.Histogram(
            x=ratings_failed_male.values,
            xbins=dict(start=0, end=10, size=1),
            marker_color=COLOR_BECHDEL_FAILED,
            opacity=0.75,
            name="Bechdel Failed",
            histnorm='percent',
            hovertemplate=(
                "Bechdel Failed - Male Director<br>"
                "Proportion: %{y:.2f}%<br>"
                f"Number of movies: {nb_movies_male_failed}<br>"
            )
        ), row=1, col=2
    )

    # Update layout for the plot
    fig.update_layout(
        title="Average Rating Depending on Bechdel Test by Director Gender",
        xaxis_title="Average Rating",
        yaxis_title="Movies [%]",
        showlegend=True,
        height=600,  # Adjust height to fit the plots
        title_x=0.5,  # Center the title
        template="plotly_white"  # Use a clean theme
    )

    # Show the plot
    fig.show()




# 3.E 2)
def revenue_bechdel(passed_male_df, failed_male_df, passed_female_df, failed_female_df):
    """
    Plot the movie box office revenue by director gender

    Args:
        df (DataFrame): Processed dataframe on which we extract the data from
        GROUP (str): "Satisfactory" or "Insufficient" group
    """
    
    legend_fem = go.Scatter(x=[None], y=[None], mode="markers", 
                         marker=dict(color=COLOR_FEMALE, size=10), name="Female Director")
    legend_male = go.Scatter(x=[None], y=[None], mode="markers", 
                        marker=dict(color=COLOR_MALE, size=10), name="Male Director")
    
    fig = go.Figure(data=[legend_fem, legend_male])
    # Male Director - Bechdel passed
    fig.add_trace(go.Violin(
        y=passed_male_df.values,
        line_color=COLOR_MALE,
        meanline_visible=True,
        box_visible=True,
        showlegend=False
    ))
    
    # Female Director - Bechdel passed
    fig.add_trace(go.Violin(
        y=passed_female_df.values,
        line_color=COLOR_FEMALE,
        meanline_visible=True,
        box_visible=True,
        showlegend=False
    ))
    
    # Male Director - Bechdel failed
    fig.add_trace(go.Violin(
        y=failed_male_df.values,
        line_color=COLOR_MALE,
        meanline_visible=True,
        box_visible=True,
        showlegend=False
    ))
    
    # Female Director - Bechdel failed
    fig.add_trace(go.Violin(
        y=failed_female_df.values,
        line_color=COLOR_FEMALE,
        meanline_visible=True,
        box_visible=True,
        showlegend=False
    ))
    
    fig.update_layout(
        title="Movie Revenue depending on Bechdel Test by Director Gender",
        xaxis=dict(
            tickmode='array',  # Define custom tick positions
            tickvals=[0.5, 2.5],  # Values where ticks appear
            ticktext=['Bechdel Passed', 'Bechdel Failed'],  # Custom tick labels
        ),
        yaxis_title="Movie Revenue",
        legend_title_text="Director Gender"
    )
    fig.show()


    # Create custom legend items for clarity
    # legend_fem = go.Scatter(x=[None], y=[None], mode="markers", 
    #                         marker=dict(color="#b7e1a1", size=10), name="Bechdel Passed")
    # legend_male = go.Scatter(x=[None], y=[None], mode="markers", 
    #                         marker=dict(color="#f29e8e", size=10), name="Bechdel Failed")

    # # Create a subplot grid with 1 row and 2 columns
    # fig = make_subplots(
    #     rows=1, cols=2,
    #     subplot_titles=["Female Directors", "Male Directors"],  # Titles for both subplots
    #     shared_yaxes=True  # Sharing the y-axis to make comparison easier
    # )

    # # Female Director - Bechdel Passed
    # fig.add_trace(go.Violin(
    #     y=passed_female_df.values,
    #     line_color="#b7e1a1",
    #     meanline_visible=True,
    #     box_visible=True,
    #     showlegend=False,  # Ensure the trace is not shown in the legend
    #     legendgroup="female_passed"  # Group the traces so that only one legend entry is shown
    # ), row=1, col=1)

    # # Female Director - Bechdel Failed
    # fig.add_trace(go.Violin(
    #     y=failed_female_df.values,
    #     line_color="#f29e8e",
    #     meanline_visible=True,
    #     box_visible=True,
    #     showlegend=False,  # Ensure the trace is not shown in the legend
    #     legendgroup="female_failed"  # Group the traces so that only one legend entry is shown
    # ), row=1, col=1)

    # # Male Director - Bechdel Passed
    # fig.add_trace(go.Violin(
    #     y=passed_male_df.values,
    #     line_color="#b7e1a1",
    #     meanline_visible=True,
    #     box_visible=True,
    #     showlegend=False,  # Ensure the trace is not shown in the legend
    #     legendgroup="male_passed"  # Group the traces so that only one legend entry is shown
    # ), row=1, col=2)

    # # Male Director - Bechdel Failed
    # fig.add_trace(go.Violin(
    #     y=failed_male_df.values,
    #     line_color="#f29e8e",
    #     meanline_visible=True,
    #     box_visible=True,
    #     showlegend=False,  # Ensure the trace is not shown in the legend
    #     legendgroup="male_failed"  # Group the traces so that only one legend entry is shown
    # ), row=1, col=2)


    # # Add the custom legend to the figure
    # fig.add_trace(legend_fem)
    # fig.add_trace(legend_male)

    # # Update layout for the plot
    # fig.update_layout(
    #     title="Movie Revenue Depending on Bechdel Test by Director Gender",
    #     xaxis=dict(
    #         tickmode='array',  # Define custom tick positions
    #         tickvals=[0.5, 2.5],  # Values where ticks appear
    #         ticktext=['Female Director', 'Male Director'],  # Custom tick labels
    #     ),
    #     yaxis_title="Movie Revenue",
    #     legend_title_text="Bechdel Test Result",
    #     height=600,  # Adjust the height for a better view
    #     template="plotly_white",  # Clean theme for better aesthetics
    #     showlegend=True  # Ensure the custom legend is displayed
    # )

    # # Show the plot
    # fig.show()


     
# 3.E 3)
def success_bechdel(df):
    """
    Plot the box office revenue in function of the average rating by director gender
    Also takes into account the budget of the movie, displaying it as the size of the dots

    Args:
        df (DataFrame): Processed dataframe on which we extract the data from
    """
    fig = px.scatter(df, x="average_rating",y="box_office_revenue", size="movie_budget", color="director_gender", 
                 title="Average Rating and Revenue of movies that pass the Bechdel Test by Director Gender",
                 color_discrete_map=COLOR_PALETTE, hover_name="movie_name",
                 labels=LABELS)
    fig.update_yaxes(title_text='Box Office Revenue')
    fig.show()
    
# # 3.E 4)
# def budget_through_years(df):
#     """
#     Plot the budget evolution accros time by movie director gender

#     Args:
#         df (DataFrame): Processed dataframe on which we extract the data from
#     """
    
#     # Plot the scatter chart
#     fig = px.scatter(df.sort_values(by='movie_release_date'), x="movie_release_date", y="movie_budget", color="director_gender", color_discrete_map=COLOR_PALETTE,
#                      title="Evolution of budget depending on movie directors through years", 
#                      hover_name="movie_name", labels=LABELS)
#     fig.show()
    
# # 3.E 4)
# def rendement_groups(df, GROUP):
#     """
#     Plot the movie rendement by director gender

#     Args:
#         df (DataFrame): Processed dataframe on which we extract the data from
#         GROUP (str): "passed" or "failed" group (output of bechdel test)
#     """
#     df = df[df['movie_rendement'] < 40]
#     fig = px.violin(df, x="director_gender", y="movie_rendement", color="director_gender", color_discrete_map=COLOR_PALETTE,
#              hover_name="movie_name", title=f"Movie Rendement of " + GROUP + " movies by Director Gender",labels=LABELS)
#     fig.show()