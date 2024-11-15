## "Her Side Story" - Beyond the Bechdel Test: Studying How Women Are Put Aside in Cinema  

### Abstract

Our project examines gender dynamics in cinema through the lens of the Bechdel Test, a metric introduced in 1985 by Alison Bechdel to assess female interaction in films. The Bechdel Test requires that a movie feature at least two named women who converse about something other than a man. While a simple benchmark, it uncovers significant gender disparities in films. This project aims to go beyond the test by exploring how women are represented through character tropes, roles, and narrative functions. We focus on how women are often sidelined, reduced to their appeareance, or defined by their relationships with men. By analyzing plot summaries and character features, this project identifies trends in female representation, tracking changes across genres and historical contexts. Our goal is to uncover patterns of gender representation over time, spark discussions on gender equality, and advocate for more inclusive and balanced storytelling.

### Research Questions

"Her Side Story" will explore the following research questions:

1. How are women sidelined in movies?
2. Does gender parity among actors influence a movie's economic performance, ratings, and global reach?
3. How do character tropes related to women affect their narrative function and presence in films over time?
4. How does the gender of a movie’s director influence the gender equity in the representation of actors, particularly women?
5. What role does the Bechdel Test play in predicting a film's representation of genders, and how does it correlate with the critical and financial success of films?

### Tools, Libraries, and Datasets

#### Tools and Libraries
- **Python Libraries**:
  - **Pandas**
  - **Numpy**
  - **Matplotlib**
  - **Seaborn** (display graphs)
  - **json** (clustering movie languages, genres and countries)
  - **tqdm** (progression bar when running functions)
  - **collections** (Counter)
  - [**Hugging Face’s transformers library**](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) (sentiment analysis)
  
- **Visualization**: Interactive visualization libraries (to be determined) will be used to create engaging data visualizations.

#### Main dataset
- **CMU Movie Summaries Dataset**: this dataset contains the following files:
  - **characters_metadata.tsv**  
  - **movie_metadata.tsv**  
  - **name_clusters.txt**  
  - **plot_summaries.txt**  
  - **tvtropes.clusters.txt**  

#### Proposed additional datasets
- [**IMDb Ratings**](https://datasets.imdbws.com/):
  - provides movie ratings and box office data
  - reduces the initial dataset size by only keeping movies whose ratings are available
- [**Bechdel Test API**](https://bechdeltest.com/api/v1/doc): This dataset
  - provides Bechdel Test result ('rating') for a group of movies.
  - drastically reduces primary dataset size
  - essential for assessing gender interaction trends and understanding the accuracy of the Bechdel Test as a predictor of gender equality in film.
- [**Gender by name - UCI** :](https://archive.ics.uci.edu/dataset/591/gender+by+name)
  - provides a wide range of first names and associated gender
  - used to recover missing gender in the character_metadata.tsv file
  - indirectly helps to analyze how gendered names correlate with character types

#### Data Handling
- Data will be extracted and pre-processed for consistency and completeness. Missing or inconsistent data will be handled through resampling, deletion, or transformation.
- We will ensure the datasets are properly aligned for analysis by standardizing key features such as character names and movie metadata. Data wrangling techniques will be employed to merge the datasets, ensuring compatibility across sources.

### Methods

#### 1. Data Handling
- **Data Wrangling**: The initial step involves cleaning and standardizing the data. First, one detects missing or inconsistent data in each dataset, by resampling or excluding incomplete (NaN) entries. The focus is set on aligning the datasets with respect to key attributes such as character tv tropes, character and actors respective names and genders, plots, and movie genres. Then, the data is further filtered in order to comply with the proposed additional datasets. This second step considerably reduces the size of the usable data, creating a more suitable dataset for our project. Finally, data is clustered ...

#### 2. Data Visualization
- **Univariable Analysis**: use of data visualisation techniques (histograms, box and scatter plots) to conduct a graphical analysis of the gender distribution of characters and actors.
- **Multivariable Analysis**: further analysis to identify relationships between various factors (e.g. the presence of female characters, movie ratings, box office performance, etc...)

#### 3. Data Description
Robust statistical methods is used to evaluate correlations, distributions, and outliers in the data, using t-tests and chi-square tests to examine the significance of the findings.

#### 4. Causal Analysis
Sensitivity analysis is performed to evaluate result uncertainty and assess model feasibility.

#### 5. Learning From Data
**Predictive Modeling**: use of linear regression models to predict gender equality in film based on certain features such as character roles (tv tropes),genres and plot summary attributes.
**Machine Learning Techniques**: techniques such as Decision Trees and Support Vector Machines (SVM) (to be determined) will be employed to create models for classifying films based on their gender representation and to predict whether a film will pass or fail the Bechdel Test.

#### 6. Sentiment Analysis
The sentiment of character descriptions and plot summaries will be analyzed using pre-trained sentiment models to assess how women’s roles are portrayed physically and emotionally.

### Timeline

**Until week 9**:
- Individual exploration and data wrangling
- Preliminary analysis on the CMU Movie Summaries dataset
- Definition of project objectives, allocation of tasks and delineation of additional datasets
**Week 10**:  
- further data wrangling
- analysis on the preprocessed data
**Week 11**:  
- Team collaboration in order refine data handling steps
- Work on initial visualizations and testing basic machine learning models
- Creation of web interface, work on storytelling and interactive features
**Week 12**:  
- Finalization of data analysis and visualizations.
- Sentiment analysis on character descriptions and plot summaries
- Further work on web interface structure, improvment of interactivness 
**Week 13**:  
- Focus on predictive modeling and refining the analysis based on feedback
- Final touches on interface
**Week 14**:  
- Completion of the final project notebook
- Focus on styling, design, and content proofreading


### Organization within the Team

1. **Until week 9**: Define clear objectives, roles, and tasks.
2. **Week 10**: Deliver the first draft of the Jupyter notebook (including key objectives and initial analysis).
3. **Week 13**: Complete 90% of the Jupyter notebook with full analysis and predictions.
4. **Week 14**: Finalize project, including interface design and interactive visualizations.

### Questions for TAs

- Are there any known issues with integrating datasets like IMDb ratings with the CMU dataset? If so, how can we address potential discrepancies?
