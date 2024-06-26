# Importing neccessary libraries
import requests
import io
import os
import sys
import logging
import shutil
from zipfile import ZipFile

# Download the .zip, extract the folder, and put the .csv file to the destination folder
def download_link(url, destination):

    # Create the path of the destination folder if it does not exist
    if not os.path.exists(destination):
        os.makedirs(destination)
        logging.info(
            f"Destination created succesfully.: {destination}")

    # HTTP request, download the .zip file
    response = requests.get(url)

    # If it is downloaded
    if response.status_code == 200:
        zip_content = io.BytesIO(response.content)

        temp_folder = os.path.join(RAW_DATA_DIR, "temp_folder")
        os.makedirs(temp_folder, exist_ok=True)

        # Extracting the .zip file
        with ZipFile(zip_content, 'r') as zip_ref:
            zip_ref.extractall(temp_folder)

        # From the extracted folder, the .csv file will be moved to the destination
        for root, dirs, files in os.walk(temp_folder):
            for file in files:
                previous_path = os.path.join(root, file)
                new_path = os.path.join(destination, file)
                shutil.move(previous_path, new_path)

        # Delete the extracted folder
        shutil.rmtree(temp_folder)
        logging.info(
            f"Download and extraction successful. File saved to {destination}")
    else:
        logging.info(
            f"Failed to download file. Status code: {response.status_code}")


#Defining variables
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

from utils import RAW_DATA_DIR, TRAIN_LINK, TEST_LINK

if __name__ == "__main__":
    print(ROOT_DIR)
    download_link(TRAIN_LINK, RAW_DATA_DIR)
    download_link(TEST_LINK, RAW_DATA_DIR)
    logging.info("Script completed successfully.")