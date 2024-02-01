import os
import logging
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


def get_project_dir(sub_dir: str) -> str:
    # Return path to a project subdirectory
    return os.path.abspath(os.path.join(os.path.dirname(__file__), sub_dir))


def makedir(destination):
    # Create destination dir
    logging.info(f"Creating destination folder...")
    if not os.path.exists(destination):
        os.makedirs(destination)
        logging.info(
            f"Destination folder created succesfully.: {destination}")
    else:
        logging.info(f"Destination folder already exists.: {destination}")
            
            
def load_file(destination: os.path) -> pd.DataFrame:
    # Load .csv file
        if os.path.exists(destination):
            logging.info(f"Loading file...: {destination}")
            return pd.DataFrame(pd.read_csv(destination))
        else:
            logging.error(f"File not exists in this destination.: {destination}")            
                
                
def save(file, destination: os.path):
    # Save .csv file
    file.to_csv(destination, index=False)
    logging.info(f"Data saved to: {destination}")                
        
        
logging.basicConfig(
    # Configure logging
    level=logging.DEBUG,
    format='%(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
        
        
        
DATA_DIR = get_project_dir(os.getenv('DATA_DIR'))

TRAIN_LINK = os.getenv('TRAIN_URL')
TEST_LINK = os.getenv('TEST_URL')

RAW_DATA_DIR = os.path.join(DATA_DIR, os.getenv('RAW_DATA_DIR'))
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, os.getenv('PROCESSED_DATA_DIR'))

RAW_TRAIN_CSV = os.path.join(RAW_DATA_DIR, os.getenv('TRAIN_CSV'))
RAW_TEST_CSV = os.path.join(RAW_DATA_DIR, os.getenv('TEST_CSV'))

PROCESSED_TRAIN_CSV = os.path.join(PROCESSED_DATA_DIR, os.getenv('TRAIN_CSV'))
PROCESSED_TEST_CSV = os.path.join(PROCESSED_DATA_DIR, os.getenv('TEST_CSV'))

OUTPUTS_DIR = get_project_dir(os.getenv('OUTPUTS_DIR'))
MODELS_DIR = os.path.join(OUTPUTS_DIR, os.getenv('MODELS_DIR'))
PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, os.getenv('PREDICTIONS_DIR'))

VALIDATION_PREDICTIONS_DIR = os.path.join(PREDICTIONS_DIR, os.getenv('VALIDATION_DIR'))
INFERENCE_PREDICTIONS_DIR = os.path.join(PREDICTIONS_DIR, os.getenv('INFERENCE_DIR'))

METRICS_TXT = os.getenv('METRICS')
PREDICTIONS_CSV = os.getenv('PREDICTIONS_CSV')

METRICS_TXT_DIR = os.path.join(PREDICTIONS_DIR, METRICS_TXT)

LOGREGMODEL_DIR = os.path.join(MODELS_DIR, 'logistic_regression.pickle')
VECTORIZER_DIR = os.path.join(MODELS_DIR, 'vectorizer.pickle')        
        