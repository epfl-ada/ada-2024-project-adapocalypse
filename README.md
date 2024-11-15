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
  - **Matplotlib**
  - **Seaborn**
  - **json**
  - **tqdm**
  - **collections**
  - **Scikit-learn**
  - **Hugging Face’s transformers library** (sentiment analysis)
  
- **Visualization**: Interactive visualization libraries (to be determined) will be used to create engaging data visualizations.

#### Main dataset
- **CMU Movie Summaries Dataset**: this dataset contains the following files:
  - **characters_metadata.tsv**  
  - **movie_metadata.tsv**  
  - **name_clusters.txt**  
  - **plot_summaries.txt**  
  - **tvtropes.clusters.txt**  
  The dataset will serve as the primary data source for our analysis. We will extract and standardize information, focusing on character roles and plot summaries.

#### Proposed additional datasets
- **IMDb Ratings**: This dataset provides movie ratings, box office data, and other metrics. It will be used to explore correlations between gender representation and film performance. It will also allow to reduce the initial dataset size, going from 81741 to ? movies.
- **Bechdel Test API**: This dataset provides information about whether films pass or fail the Bechdel Test. It is essential for assessing gender interaction trends and understanding the accuracy of the Bechdel Test as a predictor of gender equality in film. This dataset provides Bechdel Test output for ? movies, drastically reducing dataset size. 
- **Gender by name - UCI : doi : 10.24432/C55G7X - acessible at: https://archive.ics.uci.edu/dataset/591/gender+by+name : providing names (both male and female) and associated gender, this dataset is used to recover missing gender in the character_metadata.tsv file.
More globally, this dataset will help analyze how gendered names correlate with character roles, further enriching our analysis of how women are represented in film.
- **Sentiment Analysis Model**: We will use a pre-trained sentiment analysis model from Hugging Face’s transformers library to gauge the emotional tone of character descriptions and plot summaries.

#### Data Handling
- Data will be extracted and pre-processed for consistency and completeness. Missing or inconsistent data will be handled through resampling, deletion, or transformation.
- We will ensure the datasets are properly aligned for analysis by standardizing key features such as character names and movie metadata. Data wrangling techniques will be employed to merge the datasets, ensuring compatibility across sources.

### Methods

#### 1. Data Handling
- **Data Wrangling**: The initial step involves cleaning and standardizing the datasets. We will detect missing or inconsistent data, which will be addressed by resampling or excluding incomplete entries. The focus will be on aligning the datasets with respect to key attributes such as character roles, plot descriptions, and genre classifications.

#### 2. Data Visualization
- **Univariable Analysis**: We will create histograms, box plots, and scatter plots to visualize the distribution of gender roles and their characteristics in films.
- **Multivariable Analysis**: Scatter plots, line plots, and heatmaps will be used to visualize relationships between various factors, such as the presence of female characters, movie ratings, and box office performance.

#### 3. Statistical Analysis
- We will apply robust statistical methods to evaluate correlations, distributions, and outliers in the data, using t-tests and chi-square tests to examine the significance of our findings.

#### 4. Predictive Modeling
- **Linear Regression**: We will use linear regression models to predict gender equality in film based on certain features such as director gender, character roles, and plot summary attributes.

#### 5. Machine Learning Techniques
- **Decision Trees** and **Support Vector Machines (SVM)** will be employed to create models for classifying films based on their gender representation and to predict whether a film will pass or fail the Bechdel Test.

#### 6. Sentiment Analysis
- The sentiment of character descriptions and plot summaries will be analyzed using pre-trained sentiment models to assess how women’s roles are portrayed emotionally in the text.

### Timeline

**Week 9**:  
- Define project objectives, establish roles, and determine the necessary datasets.

**Week 10**:  
- Begin individual exploration and data wrangling. Start preliminary analysis on the CMU Movie Summaries dataset.

**Week 11**:  
- Team collaboration to refine data handling steps. Work on initial visualizations and testing basic machine learning models.

**Week 12**:  
- Finalize data analysis and visualizations. Begin sentiment analysis on character descriptions and plot summaries.

**Week 13**:  
- Focus on predictive modeling and refining the analysis based on feedback. Pool results for final presentation.

**Week 14**:  
- Complete the final project notebook. Focus on styling, design, and integrating interactive visualizations.

### Organization within the Team

1. **Week 9**: Define clear objectives, roles, and tasks.
2. **Week 11**: Deliver the first draft of the Jupyter notebook (including initial analysis and story narration).
3. **Week 13**: Complete 90% of the Jupyter notebook with full analysis and predictions.
4. **Week 14**: Finalize project, including presentation design, interactive visualizations, and ensure coherence across all components.

### Questions for TAs

- We’ve noticed a significant drop in the number of movies released after 2010 in our dataset. Could this be due to missing data for recent years? Would it be acceptable to exclude these years from our analysis or find supplementary sources?
- Are there any known issues with integrating datasets like IMDb ratings with the CMU dataset? If so, how can we address potential discrepancies?

---

The **Methods** and **Tools, Libraries, and Datasets** sections have been swapped, and the additional datasets are now included in the **Tools, Libraries, and Datasets** section. Let me know if further modifications are needed!
