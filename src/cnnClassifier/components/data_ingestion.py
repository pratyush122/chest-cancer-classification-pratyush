import os
import zipfile

import gdown

from cnnClassifier import logger
from cnnClassifier.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    
     
    def download_file(self)-> str:
        '''
        Fetch data from the url
        '''

        try: 
            dataset_url = self.config.source_URL
            zip_download_dir = str(self.config.local_data_file)
            os.makedirs("artifacts/data_ingestion", exist_ok=True)

            if os.path.exists(zip_download_dir):
                logger.info(f"Using existing dataset archive: {zip_download_dir}")
                return zip_download_dir

            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            downloaded_file = gdown.download(dataset_url, zip_download_dir, fuzzy=True)
            if not downloaded_file or not os.path.exists(zip_download_dir):
                raise FileNotFoundError(f"Dataset download failed: {dataset_url}")

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")
            return zip_download_dir

        except Exception as e:
            raise e
        
    
    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        expected_data_dir = os.path.join(unzip_path, "Chest-CT-Scan-data")
        if os.path.isdir(expected_data_dir) and os.listdir(expected_data_dir):
            logger.info(f"Using existing extracted dataset: {expected_data_dir}")
            return

        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        logger.info(f"Extracted dataset archive into: {unzip_path}")
