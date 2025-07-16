import os
import pandas
from src.logger import get_logger
from src.custom_exception import CustomException
import yaml
import pandas as pd # for last function load_data(path)

logger = get_logger(__name__)

def read_yaml(file_path): # creating function to read yaml file 
    # path is specified in confi->paths_config.py
    try:
        if not os.path.exists(file_path): # if file is not present
            raise FileNotFoundError(f"File is not in the given path") #using inbuilt error FileNotFoundError but we are writing our own custom exception in ()
        
        with open(file_path,"r") as yaml_file: # r is for read mode yaml_file(alias)
            config = yaml.safe_load(yaml_file)
            logger.info("Succesfully read the YAML file") # logging the success in logs folder for keeping records
            return config
    
    except Exception as e: # if any exception occurs we are doing exception handling
        logger.error("Error while reading YAML file")
        raise CustomException("Failed to read YAMl file" , e)
    
#this below mentioned code is written during 6th step data processing
def load_data(path):  #Passing the path from where we have to read the data
    try:
        logger.info("Loading data") # logging the data loading in logs folder for keeping records
        return pd.read_csv(path) # reading the data from csv file
    except Exception as e: # if any exception occurs we are doing exception handling
        logger.error(f"Error loading the data {e}") # logging the error in logs folder for keeping records
        raise CustomException("Failed to load data" , e) # raising custom exception with error message and error itself
    