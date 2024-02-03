import os
import logging
import json
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings("ignore")

CONF_FILE = os.getenv('CONF_PATH')
with open(CONF_FILE, "r") as file:
    conf = json.load(file)
    

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
        
        
        
DATA_DIR = get_project_dir(conf['general']['data_dir'])

TRAIN_LINK = conf['general']['train_url']
TEST_LINK = conf['general']['test_url']

RAW_DATA_DIR = os.path.join(DATA_DIR, conf['general']['raw_data_dir'])
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, conf['general']['processed_data_dir'])

RAW_TRAIN_CSV = os.path.join(RAW_DATA_DIR, conf['general']['train_csv'])
RAW_TEST_CSV = os.path.join(RAW_DATA_DIR, conf['general']['test_csv'])

PROCESSED_TRAIN_CSV = os.path.join(PROCESSED_DATA_DIR, conf['general']['train_csv'])
PROCESSED_TEST_CSV = os.path.join(PROCESSED_DATA_DIR, conf['general']['test_csv'])

OUTPUTS_DIR = get_project_dir(conf['general']['outputs_dir'])
MODELS_DIR = os.path.join(OUTPUTS_DIR, conf['general']['models_dir'])
PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, conf['general']['predictions_dir'])

VALIDATION_PREDICTIONS_DIR = os.path.join(PREDICTIONS_DIR, conf['general']['validation_dir'])
INFERENCE_PREDICTIONS_DIR = os.path.join(PREDICTIONS_DIR, conf['general']['inference_dir'])

METRICS_TXT = conf['general']['metrics']
PREDICTIONS_CSV = conf['general']['predictions_csv']

METRICS_TXT_DIR = os.path.join(PREDICTIONS_DIR, METRICS_TXT)

LOGREGMODEL_DIR = os.path.join(MODELS_DIR, 'logistic_regression.pickle')
VECTORIZER_DIR = os.path.join(MODELS_DIR, 'vectorizer.pickle')        

print(INFERENCE_PREDICTIONS_DIR)