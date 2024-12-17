import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import folium
import geopandas as gpd
from folium import Choropleth, CircleMarker, Popup
import pandas as pd
from plotly.subplots import make_subplots
import ast

# CONSTANT DEFINITIONS
COLOR_MALE = '#636EFA'
COLOR_FEMALE = '#EF553B'
COLOR_PALETTE = {'M': COLOR_MALE, 'F': COLOR_FEMALE}

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
          "movie_rendement":"Movie Rendement"}

# PLOTTING FUNCTIONS

# 2.A & 2.B 1)
def plot_gender_distribution(df, gender_column):
    # Exploding the gender column and calculating the counts
    gender_counts = df[gender_column].explode().value_counts()

    # Calculating percentages
    total_characters = gender_counts.sum()
    percentages = (gender_counts / total_characters) * 100

    # Create the bar chart with Plotly
    fig = px.bar(
        x=gender_counts.index,  # Gender categories
        y=gender_counts.values,  # Counts
        labels={'x': LABELS.get(gender_column), 'y': 'Number'},
        title=f"Distribution of Movie " + LABELS.get(gender_column),
        color=gender_counts.index,  # Color by gender
        color_discrete_map=COLOR_PALETTE 
    )

    # Add percentage text on top of bars
    for index, (count, percentage) in enumerate(zip(gender_counts.values, percentages)):
        fig.add_annotation(
            x=index,
            y=count + 0.05 * count,  # Shift the annotation upwards (adjust the multiplier as needed)
            text=f'{percentage:.1f}%', 
            showarrow=False, 
            font=dict(size=12), 
            align='center', 
            valign='bottom'
        )

    # Show the plot
    fig.show()
    
# 2.B 2)
def age_actors_by_dir(fig, data, title, color, dash):
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data.values,
        mode='lines',
        name=title,
        line=dict(color=color, dash=dash)
    ))
    
# 2.B 3)    
def fem_representation_by_dir(df):   
    """
    Plots the percentage of female characters in movies directed by male and female directors over time,
    including the number of movies, characters, and directors.

    Args:
    df (pd.DataFrame): DataFrame containing movie data with columns:
                       'movie_release_date', 'director_gender', 'char_F', 'char_tot', 'wikipedia_movie_id'

    Returns:
    None: Displays the plot.
    """
    # Filter movies from 1930 onwards
    df = df[df["movie_release_date"] >= 1930]

    # Split the DataFrame by director gender
    male_directors = df[df["director_gender"] == "M"]
    female_directors = df[df["director_gender"] == "F"]

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
        customdata=np.stack((male_directors.groupby("movie_release_date")["char_tot"].sum().values, 
                             movies_count_male, directors_count_male), axis=-1),
        hovertemplate=(
            "Year: %{x}<br>"
            "Proportion of Female Characters: %{y:.2f}%<br><br>"
            "Number of Characters: %{customdata[0]}<br>"
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
        yaxis_title="Percentage of Female Characters in Movies [%]",
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
    
# 2.B 4)
def movies_by_country(df):
    fig = px.bar(df, x="country", y="number_of_movies", 
                 title='Distribution of Movies by Country', labels=LABELS)
    fig.update_traces(textfont_size=10, cliponaxis=False)
    fig.show()
  
# 2.B 4)  
def map_fem_char(df, director_gender):
    """
    Generates a choropleth map of average female character percentages, 
    total characters, and movie counts in movies directed by directors of the specified gender, grouped by country.

    Parameters:
    - df: DataFrame containing movie data.
    - director_gender: Gender of the directors to filter ('M' or 'F').

    Returns:
    - Folium Map object.
    """
    # Filter the data for the specified director gender
    movies_df = df[df["director_gender"] == director_gender].copy()

    # Flatten the "movie_countries" column to the first country
    movies_df["movie_countries"] = movies_df["movie_countries"].str[0]

    # Compute percentage of female characters for each movie
    movies_df["female_percentage"] = (movies_df["char_F"] / movies_df["char_tot"]) * 100

    # Group by country to calculate metrics
    mean_female_percentage_per_country = movies_df.groupby("movie_countries")["female_percentage"].mean()
    number_movies_per_country = movies_df.groupby("movie_countries")["wikipedia_movie_id"].nunique()
    total_characters_per_country = movies_df.groupby("movie_countries")["char_tot"].sum()

    # Create a DataFrame with the results
    gender_percentages_per_country = pd.DataFrame({
        "mean_female_percentage": mean_female_percentage_per_country,
        "movies_count": number_movies_per_country,
        "total_characters": total_characters_per_country
    }).reset_index()

    # Sort data for better readability (optional)
    gender_percentages_per_country = gender_percentages_per_country.sort_values(by="mean_female_percentage", ascending=False)

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
  
# 2.B 5)
def movies_by_genre(df):
    fig = px.bar(df, x='movie_genre', y='number_of_movies', 
                 title='Distribution of Movies by Genre', 
                 color_discrete_sequence=COLOR_PALETTE, labels=LABELS)
    fig.update_traces(textfont_size=10, cliponaxis=False)
    fig.show()

# 2.B 5)
def plot_top10_genres(df, gender_real):
    genre_gender_counts = df.explode("actor_genders").explode("movie_genres")
    gender_genre_counts = genre_gender_counts.groupby(['movie_genres', 'actor_genders']).size().unstack(fill_value=0)
    gender_genre_percentages = gender_genre_counts.div(gender_genre_counts.sum(axis=1), axis=0) * 100
    genre_counts = genre_gender_counts.groupby('movie_genres').size().sort_values(ascending=False).head(10)
    gender_genre_counts_top10 = gender_genre_counts.loc[genre_counts.index]
    gender_genre_percentages_top10 = gender_genre_percentages.loc[genre_counts.index]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=gender_genre_percentages_top10.index,
        y=gender_genre_percentages_top10['F'],
        name="Female",
        marker_color="#EF553B",
    ))
    fig.add_trace(go.Bar(
        x=gender_genre_percentages_top10.index,
        y=gender_genre_percentages_top10['M'],
        name="Male",
        marker_color="#636EFA"
    ))

    fig.update_layout(
        title=f"Gender Representation Across Top-10 Movie Genres - {gender_real} Director",
        xaxis_title="Movie Genre",
        yaxis_title="Percentage of Characters (%)",
        barmode='stack',
        xaxis=dict(
            tickmode='array',
            tickvals=gender_genre_percentages_top10.index,
            ticktext=gender_genre_percentages_top10.index,
            tickangle=45
        ),
        height=600,
        legend_title="Gender",
        legend=dict(title="Gender", orientation="h", x=0.5, xanchor="center", y=1.1)
    )
    fig.show()
    
# 2.B 5)
def genres_most_fem_char(df, director_gender, sort, title):
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

# 2.B 6)
def movies_per_year(df):
    fig = px.line(df, x='movie_release_date', y='number_of_movies', 
                  title='Number of Movies Released by Year', labels=LABELS)
    fig.show()
    
def nb_char_per_movie(df, xvalues, ylabel, title, nbins, labels):
    fig = px.histogram(df, x=xvalues, title=title, nbins=nbins, labels=labels)
    fig.update_layout(
        yaxis_title=ylabel
    )
    fig.show()
    
def char_by_dir_gender(df, xvalues, ylabel, title, color, labels):
    fig = px.bar(df, x=xvalues, title=title, labels=labels, color=color)
    fig.update_layout(
        yaxis_title=ylabel
    )
    fig.show()
    
    
def nb_char_through_years(df, xvalues, title, labels):
    fig, ax = plt.subplots(figsize=(15,6))
    width = 0.4
    x = df.index  # Years
    x_male = x - width / 2
    x_female = x + width / 2

    # Plot bars side by side
    ax.bar(x_male, df[0], width, label="Male", color="gold")
    ax.bar(x_female, df[1], width, label="Female", color="royalblue")
    ax.set_title("Evolution of number of characters")
    ax.legend()
    ax.set_ylabel("Number of characters")
    ax.set_xlabel("Years")
    ax.set_xticks(x[::10])  # Adjust to show ticks every 30 years
    ax.set_xticklabels(df[xvalues][::10]);
    
def evolution_fem_char(df, xvalues, yvalues, yerr, title):
    y_err = df[yerr]

    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(df[xvalues], df[yvalues], "-o", label='Count', color="royalblue", markersize=4)
    ax.fill_between(df[xvalues], df[yvalues] - y_err, df[yvalues] + y_err, alpha=0.2, label='±1 Std. Dev.')
    ax.set_title(title)
    ax.set_xlabel("Years")
    ax.set_ylabel("Number of characters")
    ax.legend()
    
def distribution_actor_age(df, xvalues, male_label, women_label, male_avg, female_avg, title):
    fig, ax = plt.subplots()

    ax.bar(df[xvalues], df[0], label=male_label, color="gold")
    ax.bar(df[xvalues], df[1], label=women_label, color="royalblue")

    ax.axvline(male_avg, color="red", linestyle="--", linewidth=2, label=f"Avg Male Age: {male_avg:.0f}")
    ax.axvline(female_avg, color="red", linestyle="-", linewidth=2, label=f"Avg Female Age: {female_avg:.0f}")

    #get handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # #specify order of items in legend
    order = [2, 3, 0, 1]

    #add legend to plot
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    ax.set_title(title)
    ax.set_xlabel("Age")
    ax.set_ylabel("Number of actors");
    
def gender_char_types(df, ylabel, title):
    # Set up the number of subplots
    categories_per_subplot = 12
    num_subplots = (len(df) + categories_per_subplot - 1) // categories_per_subplot

    fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 4 * num_subplots))

    # Plot each subset in a separate subplot
    for i in range(num_subplots):
        start = i * categories_per_subplot
        end = start + categories_per_subplot
        subset = df.iloc[start:end]

        subset.plot(kind='bar', stacked=True, ax=axes[i], color=['royalblue', 'gold'])
        axes[i].set_title(title)
        axes[i].set_ylabel(ylabel)
        axes[i].legend(title="Gender")
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.show()
    
def dominant_emo(values, xlabel, ylabel, title):
    values.value_counts().plot(kind='bar', color='skyblue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.show()
    

def age_actors_layout(fig, xlabel, ylabel, title, legend):
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        legend_title=legend,
        template="plotly_white"
    )
    fig.show()

# Function to plot the Bechdel Test Ratings by director gender 
def bechdel_test_ratings_by_gender(df):
    # Calculate histograms for male and female directors
    male_director_hist = (
        df[df['director_gender'] == 0]['bechdel_rating']
        .value_counts(normalize=True) * 100
    )
    female_director_hist = (
        df[df['director_gender'] == 1]['bechdel_rating']
        .value_counts(normalize=True) * 100
    )

    # Get all Bechdel ratings as the x-axis
    x = df['bechdel_rating'].unique()
    # Create bar traces for Male and Female Directors
    trace_male = go.Bar(
        x=x,
        y=male_director_hist.values,
        name='Male Directors',
        width=0.4
    )
    trace_female = go.Bar(
        x=x,
        y=female_director_hist.values,
        name='Female Directors',
        width=0.4
    )
    # Create layout
    layout = go.Layout(
        title="Bechdel Test Ratings by Gender of Directors",
        xaxis=dict(title="Bechdel Rating", tickvals=x),
        yaxis=dict(title="Percentage (%)"),
        barmode='group',  # Group the bars
        bargap=0.2  # Gap between bars
    )
    # Create figure and show plot
    fig = go.Figure(data=[trace_male, trace_female], layout=layout)
    fig.show()

def corr_bechdel(df):
    # Correlation values
    correlations = df[['bechdel_rating', 'char_F', 'char_M', "director_gender"]].corr()
    bechdel_corr = correlations['bechdel_rating'].drop('bechdel_rating')

    # Plotting
    plt.figure(figsize=(8, 5))
    bechdel_corr.sort_values(ascending=True).plot(kind='barh', color='skyblue')
    plt.title('Correlation with Bechdel Rating')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Variables')
    plt.axvline(0, color='gray', linestyle='--', linewidth=1)  # Mark zero correlation
    plt.tight_layout()
    plt.show()
    
# Calculation of the average emotions for each DataFrame
def compute_mean_emotions(df):
    emotions = ["neutral", "sadness", "anger", "fear", "disgust", "surprise", "joy"]
    return df[emotions].mean()

def graph_emotions_bechdel_combined(df):
    # creation of datasets that pass or fail the bechdel test
    bechdel_complete_grade3 = df[df["bechdel_rating"]==3]
    bechdel_complete_grade012 = df[df["bechdel_rating"]!=3]

    # Creation of datasets by director gender
    bechdel_complete_grade3_men = bechdel_complete_grade3[bechdel_complete_grade3["director_gender"]==0]
    bechdel_complete_grade3_women = bechdel_complete_grade3[bechdel_complete_grade3["director_gender"]==1]
    bechdel_complete_grade012_men = bechdel_complete_grade012[bechdel_complete_grade012["director_gender"]==0]
    bechdel_complete_grade012_women = bechdel_complete_grade012[bechdel_complete_grade012["director_gender"]==1]

    # Data for the graph
    data_women_grade3 = compute_mean_emotions(bechdel_complete_grade3_women)
    data_men_grade3 = compute_mean_emotions(bechdel_complete_grade3_men)

    data_women_grade012 = compute_mean_emotions(bechdel_complete_grade012_women)
    data_men_grade012 = compute_mean_emotions(bechdel_complete_grade012_men)

    # Creation of a graph with two subplots side by side
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'polar'}, {'type': 'polar'}]],      
    )

    # First graph: For films that pass the Bechdel test (Grade 3)
    fig.add_trace(go.Scatterpolar(
        r=data_men_grade3,
        theta=data_men_grade3.index,
        fill='toself',
        name='Men - Bechdel passed',
        marker_color='gold'
    ), row=1, col=1)

    fig.add_trace(go.Scatterpolar(
        r=data_women_grade3,
        theta=data_women_grade3.index,
        fill='toself',
        name='Women - Bechdel passed',
        marker_color='royalblue'
    ), row=1, col=1)

    # Second graph: For films that do not pass the Bechdel test (Grades 0, 1, 2)
    fig.add_trace(go.Scatterpolar(
        r=data_men_grade012,
        theta=data_men_grade012.index,
        fill='toself',
        name='Men - Bechdel failed',
        marker_color='orange'
    ), row=1, col=2)

    fig.add_trace(go.Scatterpolar(
        r=data_women_grade012,
        theta=data_women_grade012.index,
        fill='toself',
        name='Women - Bechdel failed',
        marker_color='blue'
    ), row=1, col=2)

    fig.update_layout(
        title_text="Emotion Distribution by Gender for Bechdel Test Results",
        showlegend=True,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 0.25])
        ),
        template="plotly_white"
    )

    fig.show()
    
# Funtion to plot the distribution of emotions, in proportion to the scores obtained in the plot summaries
def graph_emotions(df):
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

    fig = px.pie(
        values=list(emotion_totals.values()),
        names=list(emotion_totals.keys()),
        title='Emotions in Plot Summaries',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.show()
    
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

def graph_ratio_emotion_by_director_gender(ratios_women, ratios_men):
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
        title='Ratio of Emotions by Gender',
        labels={'value': 'Ratio (%)', 'Emotion': 'Emotion'},
        barmode='group',
        color_discrete_map={
            "Women": "royalblue",
            "Men": "gold"
        }
    )

    # 
    fig.update_traces(marker=dict(opacity=0.8))
    fig.update_layout(
        xaxis_title="Emotion",
        yaxis_title="Ratio (%)",
        legend_title="Director Gender",
        template="plotly_white"
    )

    fig.show()
     
# 2.E 1)  
def avg_rating(df):
    fig = px.histogram(df, x="average_rating", color="director_gender", 
                 title="Proportional Average Rating of Movies by Director Gender", histnorm='percent',
                 barmode='group', nbins=40, color_discrete_map=COLOR_PALETTE,
                 labels=LABELS)
    fig.update_yaxes(title_text='Proportion of movies')
    fig.show()
 
 # 2.E 1)  
 
# 2.E 1)    
def avg_rating_groups(df):
    fig = px.histogram(df, x="average_rating", color="director_gender", 
                 title="Average Rating by Director Gender amongst the Optimal group", histnorm='percent',
                 barmode='group', nbins=10, color_discrete_map=COLOR_PALETTE,
                 labels=LABELS)
    fig.show()
    
# 2.E 2)
def avg_box_office(df):
    fig = px.scatter(df, x="average_rating", y='box_office_revenue', color="director_gender", color_discrete_map=COLOR_PALETTE,
                 title="Box Office Revenue in function of Average Rating by Director Gender", 
                 hover_name="movie_name", labels=LABELS)
    fig.show()
    
# 2.E 3)
def budget_through_years(df):
    fig = px.scatter(df.sort_values(by='movie_release_date'), x="movie_release_date", y="movie_budget",
            color="director_gender", hover_name="movie_name", color_discrete_map=COLOR_PALETTE,
            title="Evolution of budget depending on movie directors through years", labels=LABELS)
    fig.show()
    
# 2.E 3)
def rendement_groups(df, GROUP):
    df = df[df['movie_rendement'] < 20]
    fig = px.box(df, x="director_gender", y="movie_rendement", color="director_gender", color_discrete_map=COLOR_PALETTE,
             hover_name="movie_name", title=f"Movie Rendement of " + GROUP + " movies by Director Gender",labels=LABELS)
    fig.show()