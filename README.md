## "MADAME" - Women representation through Movie Director’s lens 

### Abstract

"What do we do now?" : this famous movie quote embodies how much women are discredited and set aside in cinema, often reduced to their appearance, or defined by their relationships with men. 
The MADAME project explores the dynamics of female representation in cinema, focusing on how the gender of a movie director influences the portrayal of women. By examining movie genres, character tropes, plot emotions and success, the project investigates how different male and female movie directors depict women. Through an in-depth analysis of character attributes and plot summaries, MADAME project identifies trends in gender representation, for both male and female directed movies. Readers will follow the story of Madame, who leads the analysis, uncovering insights and sparking discussions on the way male and female directors actually represent women in movies. Our work aims to question the rooted stereotypes in film industry to advocate for more inclusive and diverse narratives in film. In the end, Madame finds out that female directors depict more women in their movies, though usually using the same clichés as male directors. 

### Research Questions

- How does the **gender of a movie director** influence the portrayal of women in cinema?

- What are the **key female stereotypes** in movies?

- Are female director **better than men** at depicting women in movies? 

### Tools, Libraries, and Datasets

#### Tools and Libraries
- **Python Libraries**:
  - **Pandas**
  - **Numpy**
  - **Plotly** (majority of the visualization)
  - **Sklearn** (machine learning processing)
  - **Matplotlib** (some graphs)
  - **Folium and geopandas** (display maps)
  - **Json** (clustering movie languages, genres and countries)
  - **Tqdm** (progression bar when running functions)
  - **Collections** (Counter)
  - **Ast** (string transformation)
  - **Seaborn** (display correlation matrix)
  - [**Hugging Face’s transformers library**](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) (sentiment analysis)


#### Main dataset
- **CMU Movie Summaries Dataset**: contains the following files:
  - **characters_metadata.tsv**  
  - **movie_metadata.tsv**  
  - **plot_summaries.txt**  (sentiment analysis using BERT Transformers)
  - **tvtropes.clusters.txt**  (was replaced by a more detailed dataset)
  - **name_clusters.txt**  (was never used)

#### Additional datasets
- [**Gender by name - UCI** :](https://archive.ics.uci.edu/dataset/591/gender+by+name)
  - provides a wide range of first names and associated gender
  - used to recover missing gender for the characters in the character_metadata.tsv file and for the directors, first doing wikipedia webscrapping to get their name
  - poses the foundation for our analysis about gender

- [**TV Tropes - Gender Bias**](https://aclanthology.org/2020.nlpcss-1.23/): 
  - provides a more complete TV tropes dataset (from 500 to 6000 movies)
  - essential to attribute a specific gender to a trope and thus analyze their stereotypes


- [**Bechdel Test API**](https://bechdeltest.com/api/v1/doc): 
  - provides Bechdel Test result ('rating') for a group of movies
  - drastically reduces primary dataset size
  - essential for assessing gender interaction trends and understanding the accuracy of the Bechdel Test as a predictor of gender equality in film
  - directly linked to our question concerning female stereotypes

- [**IMDb Ratings**](https://datasets.imdbws.com/):
  - provides movie ratings data
- [**TMDB Success**](https://www.kaggle.com/datasets/juzershakir/tmdb-movies-dataset):
  - provides box office and movie budget data



## Repository Structure

```sh
├── .gitignore                              # Git ignore file 
├── data
│ ├── processed                             # processed data
│ │   └── transitionary                      
│ │       ├── (bechdel_ratings.csv)  │             
│ │   │    ├── (characters_metadata.csv) 
│ │   │    ├── (imdb_ratings.csv) 
│ │   │    ├── (movies_director.csv) 
│ │   │    ├── (movies_metadata.csv) 
│ │   │    ├── (movies_success.csv) 
│ │   │    ├── (plot_emotions.csv) 
      │── (movies_complete.csv) 
│ └── raw                                   # raw data
│     ├── (character.metadata.tsv)
│     ├── (clusters.json) 
│     ├── (movie.metadata.tsv) 
│     ├── (name.clusters.txt) 
│     ├── (plot_summaries.txt) 
│     ├── (README.txt) 
│     └── (tvtropes.clusters.txt) 
├── src                                     # Source code 
│   ├── data                                # data processing 
│   │   ├── external_data # to store big files that are ignored
│   │   ├── data_cleaner.py 
│   │   ├── data_loader.py 
│   │   └── data_transformer.py 
│   └── utils                               # utility functions 
│       ├── model_ML # to store ML model for easy re-use
│       ├── methods.py 
│       └── visualization.py 
├── results.ipynb                           # results and analysis notebook 
│ 
├── pip_requirements.txt                    # pip requirements file
└── README.md 
```

## Methods

### Management of External Datasets

Several large datasets essential for the MADAME project were excluded from the Git repository and added to `.gitignore`. These files are located in the `src/data/external_data` directory and need to be downloaded manually from their respective sources. Below is the list of datasets used:

1. **`title.basics.tsv`**  
   - **Source**: IMDB database  
   - **Description**: Contains metadata about films, including titles, release years, and genres.  

2. **`title.ratings.tsv`**  
   - **Source**: IMDB database  
   - **Description**: Provides information about movie ratings and vote counts.  

3. **`TMDB_movie_dataset_v11.csv`**  
   - **Source**: [Kaggle](https://www.kaggle.com/)  
   - **Description**: A comprehensive dataset with detailed information about movies, such as budgets, revenues, and more.  

4. **`film_tropes.csv`**  
   - **Source**: [TV Tropes Repository](https://github.com/dhruvilgala/tvtropes?tab=readme-ov-file)  
   - **Description**: Includes data on film tropes and their associations.  

5. **`genderedness_filter.csv`**  
   - **Source**: [TV Tropes Repository](https://github.com/dhruvilgala/tvtropes?tab=readme-ov-file)  
   - **Description**: Provides insights into gender-related classifications of tropes.  

#### Data Handling & preprocessing
- **Data Wrangling**: extraction and cleaning of the data
- Focus on aligning the datasets with respect to key attributes such as title of the movie, character tv tropes, character and actors respective names and genders, plots, and movie genres
- Data filtering to comply with the proposed additional datasets and assure compatibility across sources
- Clustering of the movie genres and languages to handle bigger and meaningful classes 

#### Data Visualization
- **Univariable Analysis**: use of data visualisation techniques (histograms, scatter plots, violin plots...) to conduct a graphical analysis of the gender distribution of characters and actors.

- **Multivariable Analysis**: further analysis to identify relationships between various factors (e.g. the presence of female characters, movie ratings, box office performance, etc...) using bubble charts, map, race charts...

#### Data Description
Robust statistical methods is used to evaluate correlations, distributions, and outliers in the data. Pearson and Spearman correlations were used (see the correlation between rating/revenue). Chi-square tests were conducted (director gender/bechdel test outcome).

#### Learning From Data
- **Machine Learning Techniques**: a logistic regression was employed to predict whether a film will pass or fail the Bechdel Test based on our data, not knowing the dialogues. An accuracy of 66% was obtained on the test set.

#### Sentiment Analysis
The sentiment of plot summaries was analyzed using BERT Transformers to assess if the director gender had an impact on the overall emotion of its movie.

### Timeline

**Week 1 to 9**:
  - Individual exploration and data wrangling
  - Preliminary analysis on the CMU Movie Summaries dataset
  - Definition of project objectives, allocation of tasks and delineation of additional datasets
  - First graphs on movies and character metadata, to understand the dataset
  - Finding Bechdel dataset, IMDB ratings

**Week 10**:  
  - Definition of our topic and questions: female VS male directors to depict women
  - Secondary and more detailed graphs, linked to our topic
  - Sentiment analysis on character descriptions and plot summaries
  - Finding TV tropes dataset, TMDB success

**Week 11**:  
  - Team collaboration in order to refine data handling steps
  - Work on initial visualizations and analysis
  - Discovering Plotly
  - Creation of the Madame data story

**Week 12**:  
  - Finalization of data analysis and visualizations
  - Conducting statistical tests on our analysis
  - Website creation, using svelte UI framework 
  - Data and visualization formatting in json

**Week 13**:  
  - Focus on predictive modeling and refining the analysis based on feedback
  - Further work on web interface structure, improvement of interactiveness 
  - Further work on data and visualization formatting in json
  - Repository ‘cleaning’ (restructuring git and organizing functions, results files etc…) 
  - Storytelling writing

**Week 14**:  
  - Completion of the final project notebook
  - Final work on data and visualization formatting in json 
  - Storytelling implementation on the webpage 
  - Styling and design of the webpage
  - Editing the readme 
  - Content proofreading 

### Organization within the team

| Member | Tasks |
| --- | --- |
| Coralie | General analysis, focus on TV tropes and movies / Formatting data and visualizations in json / Website development & design |
| Maximilien | TV tropes / plot summaries and Transformers sentiment analysis / Bechdel ML model |
| Juliette | Movie Success analysis / Data preprocessing / Project timeline management / Repository structure |
| Mahlia | General analysis, focus on TV tropes and characters / Bechdel ML model / Data preprocessing / Repository organisation and coordination |
| Pernelle | General analysis / Storytelling managment / Website graphic design / Readme managment |

First, to seek through the data, we decided to split the work according to the different datasets. While Coralie, Pernelle and Mahlia were assigned to the "movie_metadata" and "character_metadata" datasets, Maximilien was in charge of "tv_tropes" and "plot_summaries", Juliette worked on the "IMDb" ans "TMDB" datasets. 

Then, Juliette and Mahlia carried the data preprocessing, and the formatting of the additional datasets, repository organisation and result analysis within it. Maximilien worked on methods functions and results analysis, and Coralie and Pernelle focused on developping and designing the webiste, as well as storytelling. 
To follow, Mahlia and Maximilien created a Machine Learning Model to predict the output of the Bechdel Test, given several movie features. 

Finally, everyone participated in creating visualizations and graphs and respective discussion of the results. 