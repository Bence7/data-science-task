# Importing neccessary libraries
import os
import logging
import pickle
import sys
import time
import json
import numpy as np

from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Load the variables from the .env file into the environment
load_dotenv()

# To import functions, variables from utils.py, the path needs to be added to the system path
sys.path.append(os.path.dirname(os.path.abspath('utils.py')))
from utils import VALIDATION_PREDICTIONS_DIR, makedir, load_file, MODELS_DIR, RAW_TRAIN_CSV, PROCESSED_DATA_DIR, PROCESSED_TRAIN_CSV, RAW_TEST_CSV, PROCESSED_DATA_DIR, PROCESSED_TEST_CSV

# To import functions, variables from evaluation.py and data_preparation.py, the path needs to be added to the system path    
sys.path.append(os.path.join(os.path.dirname(os.path.abspath('src')), 'src'))
from data_preparation import DataPreparation
from evaluation import make_predictions, evaluate

# To import hyperparameters 
CONF_FILE = os.getenv('CONF_PATH')
with open(CONF_FILE, "r") as file:
    conf = json.load(file)


class Training:
    def __init__(self) -> None:
        # Initialize the model with the best parameters 
        self.model = LogisticRegression(
            C=conf["lr"]["C"], penalty=conf["lr"]["penalty"], solver=conf["lr"]["solver"])
        self.vectorizer = TfidfVectorizer()

    def run_training(self, file_destination: os.path, folder_destination: os.path, model_destination: os.path) -> None:
        """
        Training pipeline

        Args:
            file_destination (os.path): Destination of the file, the model will be trained on this file.
            folder_destination (os.path):  Destination, where the predictions, metrics will be saved.
            model_destination (os.path): Destination of the model.
        """
        # Load the training file.
        dataframe = load_file(file_destination)
        
        # Split the file.
        X_train, X_test, y_train, y_test = self._train_test_split(
            dataframe['review'], dataframe['sentiment'])
        
        # Vectorize the file.
        X_train_tfid, X_test_tfid = self._vectorize(X_train, X_test, model_destination)
        
        # Train the model on the splitted file.
        self._train(X_train_tfid, y_train.values)
        
        # Make folder where the predictions, metrics will be saved.
        makedir(folder_destination)
        
        # Making predictions and saving them.
        y_preds = make_predictions(self.model, folder_destination, X_test_tfid)
        
        # Making metrics and saving them.
        evaluate(y_test, y_preds, folder_destination)
        
        # Save the model.
        self._save_model(model_destination)


    def _train_test_split(self, df, labels):
        # Split the data set into train, and validation.
        logging.info("Split data set...")
        return train_test_split(df, labels, train_size=conf['general']['train_size'], random_state=conf['general']['random_state'])


    def _vectorize(self, train: np.array, test: np.array, model_destination: os.path):
        logging.info("Start vectorizing...")
        
        # Vectorize X_train 
        X_train_tfid = self.vectorizer.fit_transform(train)
        
        # Transform the X_test
        X_test_tfid = self.vectorizer.transform(test)
        
        # Save the vectorizer, it will be used when vectorizing inference
        self._save_vectorizer(model_destination)
        logging.info("Vectorizing completed...")
        return X_train_tfid, X_test_tfid


    def _save_vectorizer(self, model_destination: os.path):
        # Save the vectorizer, it will be used when vectorizing inference
        makedir(model_destination)
        with open(os.path.join(model_destination, 'vectorizer.pickle'), 'wb') as file:
            pickle.dump(self.vectorizer, file)


    def _train(self, X_train: np.array, y_train: np.array):
        # Fitting the model
        logging.info("Start training...")
        start_time = time.time()
        self.model.fit(X_train, y_train)
        end_time = time.time()
        logging.info(f"Training completed in {end_time - start_time} seconds.")


    def _save_model(self, model_destination: os.path):
        # Save the fitted model on the destination
        makedir(model_destination)
        with open(os.path.join(model_destination, 'logistic_regression.pickle'), 'wb') as file:
            pickle.dump(self.model, file)


if __name__ == "__main__":
    dp = DataPreparation()
    #Prepare train_csv file
    dp.preparation(RAW_TRAIN_CSV, PROCESSED_DATA_DIR, PROCESSED_TRAIN_CSV)
        
    #Prepare test_csv file
    dp.preparation(RAW_TEST_CSV, PROCESSED_DATA_DIR, PROCESSED_TEST_CSV)

    #Train model on train_csv file    
    tr = Training()
    tr.run_training(PROCESSED_TRAIN_CSV, VALIDATION_PREDICTIONS_DIR, MODELS_DIR)
