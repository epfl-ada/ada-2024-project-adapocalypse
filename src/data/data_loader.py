import pandas as pd
import os
import requests
from bs4 import BeautifulSoup
    
def load_csv(file_path, is_tsv=False, has_column_names=True,  column_names=None):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    try:
        if (is_tsv):
            df = pd.read_csv(file_path, sep='\t')
        else:
            df = pd.read_csv(file_path)
        print(f"Loaded data from {file_path}, shape: {df.shape}")
        if not has_column_names:
            df.columns = column_names
        return df
    except Exception as e:
        raise RuntimeError(f"Error loading CSV file: {e}")

def load_txt(file_path, column_names=None):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    try:
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                row = line.strip().split('\t')
                data.append(row)
        df = pd.DataFrame(data)
        # add column names
        df.columns = column_names
        print(f"Loaded data from {file_path}")
        return df
    except Exception as e:
        raise RuntimeError(f"Error loading TXT file: {e}")
    

def load_movies_director(DATA_FOLDER_PATH):
    movies_director_df = pd.read_csv(DATA_FOLDER_PATH+'movies_director.csv', header=None, 
                    names=['wikipedia_movie_id', 'director_name', 'gender'])
    return movies_director_df[1:]

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