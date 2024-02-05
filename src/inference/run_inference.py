# Importing neccessary libraries
import os
import pickle
import sys
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# Load the variables from the .env file into the environment
load_dotenv()

# To import functions, variables from utils.py, the path needs to be added to the system path
sys.path.append(os.path.dirname(os.path.abspath('utils.py')))
from utils import load_file, makedir, INFERENCE_PREDICTIONS_DIR, LOGREGMODEL_DIR, VECTORIZER_DIR, PROCESSED_TEST_CSV
    
# To import functions, variables from evaluation.py, the path needs to be added to the system path    
sys.path.append(os.path.join(os.path.dirname(os.path.abspath('src')), 'src'))
from evaluation import evaluate, make_predictions


class RunInference():
    def __init__(self):
        self.model = None
        self.vectorizer = None
        
        
    def run(self, file_destination: os.path, model_destination: os.path, vectorizer_destination: os.path, folder_destination: os.path):
        """
        Run inference pipeline.

        Args:
            file_destination (os.path): Destination of the inference file.
            model_destination (os.path): Destination of the model.
            vectorizer_destination (os.path): Destination of the vectorizer.
            folder_destination (os.path): Destination, where the predictions, metrics will be saved.
        """
        # Load the model.
        self.model = self._load_model(model_destination)
        
        # Load vectorizer.
        self.vectorizer = self._load_model(vectorizer_destination)
        
        # Load the inference file.
        test_csv = load_file(file_destination)
        
        # Transform the inference file with the vectorizer.
        test_tfid = self.vectorizer.transform(test_csv['review'])
        
        # Make the folder, where the predictions, metrics will be saved.
        makedir(folder_destination)
        
        # Making predictions.
        predictions = make_predictions(self.model, folder_destination, test_tfid)
        y_true=test_csv['sentiment'].values
        
        # Evaluate and saving the predictions, metrics.
        evaluate(predictions, y_true, folder_destination)
        
    
    
    def _load_model(self, dir: os.path) -> LogisticRegression() or TfidfVectorizer():
        #Load the model or the vectorizer file.
        with open(dir, 'rb') as file:
            return pickle.load(file)
            
                    
if __name__ == "__main__":
    # Fitted model make predictions on inference file.
    ri = RunInference()
    ri.run(PROCESSED_TEST_CSV, LOGREGMODEL_DIR, VECTORIZER_DIR, INFERENCE_PREDICTIONS_DIR)