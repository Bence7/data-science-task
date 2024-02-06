# Importing neccessary libraries
import os
import sys
import logging
import re
import time

# Basic data preparing libraries
from html import unescape
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import  word_tokenize

# To import functions, variables from utils.py, the path needs to be added to the system path
sys.path.append(os.path.dirname(os.path.abspath('utils.py')))
from utils import makedir, load_file, save

# Load the variables from the .env file into the environment
load_dotenv()

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

#Initialize stop words with english words
stop_words = set(stopwords.words('english'))


# Data preparation class
class DataPreparation:
    def __init__(self) -> None:
            None
            

    def preparation(self, raw_destination: os.path, folder_destination: os.path, processed_destination: os.path):
        """
        Data preparation pipeline.

        Args:
            raw_destination (os.path): The destination of the raw file, that will be prepared.
            folder_destination (os.path): The destination folder, where the prepared file will be saved.
            processed_destination (os.path): The destination of the preprocessed file. Contains the name of the file.
        """
        # Load the raw file.
        dataframe = load_file(raw_destination)
        logging.info("Preparing data...")
        start_time = time.time()
        
        # Clean the data (NLP).        
        dataframe['review'] = dataframe['review'].apply(self._prepare_data)
        
        # Encoding the 'sentiment' feature.
        dataframe['sentiment'] = dataframe['sentiment'].apply(self._encoding_sentiment)
        end_time = time.time()
        logging.info(f"Data preparing completed in {end_time - start_time} seconds.")
        
        # Creating the folder where the processed dataframe will be saved.
        makedir(folder_destination) 
        
        # Saving the processed dataframe.
        save(dataframe, processed_destination)
             

    def _prepare_data(self, text):
        """
        Basic NLP features.

        Args:
            text (string): This will be prepared.


        Returns:
            string: Returning string after applying basic NLP features.
        """
        # Remove HTML tags.
        text = self._clean_html_tags(text)
        
        #Remove punctuations and numbers.
        text = re.sub('[^a-zA-Z]',' ', text)
        
        # Change uppercase to lowercase.
        text = str(text).lower()
        
        # Tokenize the text.
        tokenized = word_tokenize(text)
        
        # Remove common english stopwords.
        tokenized = [item for item in tokenized if item not in stop_words]
        
        # Lemmatize the words.
        tokenized = [lemmatizer.lemmatize(word=w, pos='v') for w in tokenized]
        
        # Remove word that is shorter than three letter.
        tokenized = [i for i in tokenized if len(i) > 2]
        
        # Concatenate the words into string.
        text = ' '.join(tokenized)
        
        return text
    

    # Clean text from HTML tags.
    def _clean_html_tags(self, html_text):
        soup = BeautifulSoup(html_text, 'html.parser')
        cleaned_text = soup.get_text(separator=' ', strip=True)
        cleaned_text = unescape(cleaned_text)
        return cleaned_text    
    
    
    # Encoding the 'sentiment' feature.
    def _encoding_sentiment(self, text):
        if text == 'positive':
            return 1
        else: 
            return 0
        
        

