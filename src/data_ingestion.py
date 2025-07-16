import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        self.config = config["data_ingestion"]
        self.train_test_ratio = self.config["train_ratio"]

        # Path from config.yaml (improved clarity)
        self.local_file_path = self.config["local_file_path"]

        # Create raw directory if not exists
        os.makedirs(RAW_DIR, exist_ok=True)
        logger.info(f"Data Ingestion initialized with local file path: {self.local_file_path}")

    def load_csv_from_local(self):
        try:
            # Check if file exists and copy to RAW_FILE_PATH
            if os.path.exists(self.local_file_path):
                logger.info("Found local CSV file. Copying to raw data path...")
                with open(self.local_file_path, 'rb') as src_file:
                    with open(RAW_FILE_PATH, 'wb') as dest_file:
                        dest_file.write(src_file.read())
                logger.info(f"Copied local file to {RAW_FILE_PATH}")
            else:
                raise FileNotFoundError(f"Local file not found at {self.local_file_path}")
        except Exception as e:
            logger.error("Error while loading CSV from local")
            raise CustomException("Failed to load local CSV file", e)

    def split_data(self):
        try:
            logger.info("Starting data split process")
            data = pd.read_csv(RAW_FILE_PATH)

            train_data, test_data = train_test_split(
                data,
                test_size=1 - self.train_test_ratio,
                random_state=42
            )

            train_data.to_csv(TRAIN_FILE_PATH)
            test_data.to_csv(TEST_FILE_PATH)

            logger.info(f"Train data saved to {TRAIN_FILE_PATH}")
            logger.info(f"Test data saved to {TEST_FILE_PATH}")
        except Exception as e:
            logger.error("Error while splitting data")
            raise CustomException("Failed to split data into training and test sets", e)

    def run(self):
        try:
            logger.info("Starting full data ingestion process")

            self.load_csv_from_local()
            self.split_data()

            logger.info("Data ingestion completed successfully")
        except CustomException as ce:
            logger.error(f"CustomException: {str(ce)}")
        finally:
            logger.info("Data ingestion completed")


if __name__ == "__main__":
    # Load YAML config before passing to class
    try:
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)

        data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
        data_ingestion.run()

    except Exception as e:
        logger.error("Failed to run data ingestion due to config loading issue")
        raise CustomException("Failed to load config or run ingestion", e)
