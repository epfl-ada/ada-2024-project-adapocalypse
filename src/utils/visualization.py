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
from sklearn.metrics import confusion_matrix
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# CONSTANT DEFINITIONS
COLOR_MALE = '#2D9884'
COLOR_FEMALE = '#6E17C6'
COLOR_MALE_LIGHT = '#17D07D'
COLOR_FEMALE_LIGHT = '#B56BEA'
COLOR_NEUTRAL = '#FDE047'
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

GENRE = "movie_genre"
NB_MOVIES = "number_of_movies"
COUNTRY = "country"
RELEASE_DATE = "movie_release_date"
DIR_GENDER = "director_gender"
ACT_GENDERS = "actor_genders"

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
    fig = px.bar(df, x=COUNTRY, y=NB_MOVIES, color_discrete_sequence=[COLOR_NEUTRAL],
                 title='Distribution of Movies by Country', labels=LABELS)
    fig.update_traces(textfont_size=10, cliponaxis=False)
    fig.show()
    
# 2
def movies_by_genre(df):
    """
    Plot the distribution of movies by genre

    Args:
        df (DataFrame): Processed datafram on which we extract the data from
    """
    fig = px.bar(df, x=GENRE, y=NB_MOVIES, color_discrete_sequence=[COLOR_NEUTRAL], 
                 title='Distribution of Movies by Genre', labels=LABELS)
    fig.update_traces(textfont_size=10, cliponaxis=False)
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
    plt.hist(df[RELEASE_DATE], bins=range(df[RELEASE_DATE].min(), df[RELEASE_DATE].max() + ONE), color=COLOR_NEUTRAL) # add one to max to include it
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
    
    label = "Movie Characters" if gender_column == ACT_GENDERS else LABELS.get(gender_column) 

    # Create the bar chart with Plotly
    fig = px.bar(
        x=gender_counts.index,  # Gender categories
        y=gender_counts.values,  # Counts
        labels={'x': label, 'y': 'Number'},
        title=f"Gender Distribution of {label}",
        color=gender_counts.index,  # Color by gender
        color_discrete_map=COLOR_PALETTE 
    )
    
    fig.update_layout(legend_title_text="Gender")

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
        line=dict(color=color, dash=dash)
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
        yaxis_title='Percentage (%)',
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
    df = df[df[RELEASE_DATE] >= 1930]

    # Split the DataFrame by director gender
    male_directors = df[df[DIR_GENDER] == MALE]
    female_directors = df[df[DIR_GENDER] == FEMALE]

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
        yaxis_title="Percentage of Female Characters in Movies [%]",
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
def map_fem_char(df, director_gender):
    """
    Generates a choropleth map of average female character percentages, 
    total characters, and movie counts in movies directed by directors of the specified gender, grouped by country

    Parameters:
    - df: Processed dataframe on which we extract the data from
    - director_gender: Gender of the directors to filter ('M' or 'F')

    Returns:
    - Folium Map object
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
    df["movie_countries"] = df["movie_countries"].replace(country_mapping)

    # Merge gender data with GeoPandas world map
    world = world.merge(df, left_on="name", right_on="movie_countries", how="left")

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
        data=df,
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
            
    return 

# 3.B 5)
def plot_top10_genres(df, gender_real):
    """
    Plot the gender representation across the 10 Movie Genres most represented

    Args:
        df (DataFrame): Processed dataframe on which we extract the data from
        gender_real (str): Gender of the movie director
    """

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df.index,
        y=df[FEMALE],
        name=FEMALE,
        marker_color=COLOR_FEMALE,
    ))
    fig.add_trace(go.Bar(
        x=df.index,
        y=df[MALE],
        name=MALE,
        marker_color=COLOR_MALE
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
        legend_title="Gender",
        legend=dict(title="Gender", orientation="h", x=0.5, xanchor="center", y=1.1)
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
    
# 3.C 1)
def bechdel_test_ratings_by_gender(df):
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
        width=0.4
    )
    trace_female = go.Bar(
        x=x,
        y=[female_director_hist.get(label, 0) for label in x],
        name='Female Directors',
        marker_color=COLOR_FEMALE,
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

# 3.C 1)
def corr_bechdel(df):
    """
    Plot the correlation of the Bechdel ratings with the number of men and women characters and the gender director.
    
    Args: 
        df (DataFrame): Processed dataframe on which we extract the data from
    """
    # Correlation values
    correlations = df[['bechdel_rating', 'char_F', 'char_M', "director_gender"]].corr()
    bechdel_corr = correlations['bechdel_rating'].drop('bechdel_rating')

    # Plotting
    plt.figure(figsize=(8, 5))
    bechdel_corr.sort_values(ascending=True).plot(kind='barh', color=COLOR_NEUTRAL)
    plt.plot(color=COLOR_NEUTRAL)
    plt.title('Correlation with Bechdel Rating')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Variables')
    plt.axvline(0, color='grey', linestyle='--', linewidth=1)  # Mark zero correlation
    plt.tight_layout()
    plt.show()
    
# 3.C 2)
def graph_emotions(emotion_totals):
    """
    Plot the emotions distribution in the plot summaries.

    Args: 
        emotion_totals (dict): Processed dictionary on which we extract the data from

    """

    fig = px.pie(
        values=list(emotion_totals.values()),
        names=list(emotion_totals.keys()),
        title='Emotions in Plot Summaries',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.show()
    
# 3.C 2)
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
    bechdel_grade3_men = bechdel_grade3[bechdel_grade3["Gender"]==0]
    bechdel_grade3_women = bechdel_grade3[bechdel_grade3["Gender"]==1]
    bechdel_grade012_men = bechdel_grade012[bechdel_grade012["Gender"]==0]
    bechdel_grade012_women = bechdel_grade012[bechdel_grade012["Gender"]==1]

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
    
# 3.C 2)
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
        title='Ratio of Emotions by Gender',
        labels={'value': 'Ratio (%)', 'Emotion': 'Emotion'},
        barmode='group',
        color_discrete_map={
            "Women": COLOR_FEMALE,
            "Men": COLOR_MALE
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

# 3.C 2)
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
        marker=dict(size=6)
    ))
    # Male directors
    fig.add_trace(go.Scatterpolar(
        r=values_men,
        theta=emotions,
        fill='toself',
        name='Male Directors',
        line=dict(color=COLOR_MALE),
        marker=dict(size=6)
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
    
# 3.C 3)
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
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.title('Normalized Random Forest Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# 3.D
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
        marker=dict(colors=[COLOR_PALETTE]),
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
        marker=dict(colors=[COLOR_PALETTE]),
        name="Female directors",
        hovertemplate='%{label}: %{value:.2f}%'  # Customize hovertemplate (show label and percentage with 2 decimals)
    ))

    # Update layout for displaying both charts side by side and proper annotations
    fig.update_layout(
        title="Representation of Male/Female Tropes by Director Gender",
        grid=dict(rows=1, columns=2),
        annotations=[
            dict(text='Male Directors', x=0.1, y=0.5, font_size=12, showarrow=False),
            dict(text='Female Directors', x=0.6, y=0.5, font_size=12, showarrow=False)
        ],
        showlegend=True,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.2)
    )

    fig.show()
   
# 3.D
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
def avg_rating_groups(df):
    """
    Plot the average rating of movies categorized as Optimal/Worst by gender of the movie director

    Args:
        df (DataFrame): Processed dataframe on which we extract the data from
    """
    fig = px.histogram(df, x="average_rating", color="director_gender", 
                 title="Average Rating by Director Gender amongst the Optimal group", histnorm='percent',
                 barmode='group', nbins=10, color_discrete_map=COLOR_PALETTE,
                 labels=LABELS)
    fig.show()
    
# 3.E 2)
def avg_box_office(df):
    """
    Plot the box office revenue in function of the average rating by director gender

    Args:
        df (DataFrame): Processed dataframe on which we extract the data from
    """
    fig = px.scatter(df, x="average_rating", y="box_office_revenue", color="director_gender", color_discrete_map=COLOR_PALETTE,
                     title="Box Office Revenue in function of Average Rating by Director Gender", 
                     hover_name="movie_name", labels=LABELS)
    fig.show()
    
# 3.E 3)
def budget_through_years(df):
    """
    Plot the budget evolution accros time by movie director gender

    Args:
        df (DataFrame): Processed dataframe on which we extract the data from
    """
    
    # Plot the scatter chart
    fig = px.scatter(df.sort_values(by='movie_release_date'), x="movie_release_date", y="movie_budget", color="director_gender", color_discrete_map=COLOR_PALETTE,
                     title="Evolution of budget depending on movie directors through years", 
                     hover_name="movie_name", labels=LABELS)
    fig.show()
    
# 3.E 3)
def rendement_groups(df, GROUP):
    """
    Plot the movie rendement by director gender

    Args:
        df (DataFrame): Processed dataframe on which we extract the data from
        GROUP (str): "passed" or "failed" group (output of bechdel test)
    """
    df = df[df['movie_rendement'] < 40]
    fig = px.violin(df, x="director_gender", y="movie_rendement", color="director_gender", color_discrete_map=COLOR_PALETTE,
             hover_name="movie_name", title=f"Movie Rendement of " + GROUP + " movies by Director Gender",labels=LABELS)
    fig.show()