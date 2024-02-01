# Importing neccessary libraries
import logging
import os
import sys
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# To import functions, variables from utils.py, the path needs to be added to the system path
sys.path.append(os.path.dirname(os.path.abspath('utils.py')))
from utils import METRICS_TXT, PREDICTIONS_CSV, save


''' A function that makes predictions.
    The 'model' will make predictions from the 'test' array, 
    and the result will be saved to the 'destination' folder.'''
def make_predictions(model: LogisticRegression(), destination: os.path, test: np.array) -> np.array:
    logging.info("Making predictions...")
    # Making predictions
    y_preds = model.predict(test)
    
    # Saving the predictions to the destination
    save(pd.DataFrame({'predictions': y_preds}),
         os.path.join(destination, PREDICTIONS_CSV))

    return y_preds


''' An evaluation function.
    It has two parameters. It will calculate basic classification metrics from the 'y_true' and 'predictions'.
    Result will be saved to the 'destination' folder.'''
def evaluate(y_true: np.array, predictions: np.array, destination: os.path):
    logging.info("Evaluate...")
    
    #Making the report
    report = classification_report(y_true=y_true, y_pred=predictions)
    
    #Save the report to the destination
    with open(os.path.join(destination, METRICS_TXT), 'w') as file:
        file.write(report)