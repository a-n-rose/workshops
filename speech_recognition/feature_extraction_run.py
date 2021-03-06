'''
Extracts speech features and puts into SQL database

I want to see if 

MFCC vs FBANK
dominant frequency VS fundamental frequency

work better/worse for 1) speech recognition 2) gender classification 3) handling noise
'''
import time
import os
from sqlite3 import Error
import pandas as pd

import user_input 
from errors import ExitApp, FeatureExtractionError
import feature_extraction_functions as featfun 

import logging
from my_logger import start_logging, get_date
logger = logging.getLogger(__name__)


def main(script_purpose,database=None,feature_type=None,num_features=None,noise=False,label_column=None,label_data_type=None):
    current_filename = os.path.basename(__file__)
    session_name = get_date() #make sure this session has a unique identifier - link to model name and logging information
    
    #set default values
    if database is None:
        database = "speech_features.db"
    if feature_type is None:
        feature_type = "fbank_delta"
    if num_features is None:
        num_features = 40
    if label_column is None:
        label_column = "word"
    if label_data_type is None:
        label_data_type = "TEXT"
    
    start = time.time()
    
    try:
    
        start_logging(script_purpose)
        logging.info("Running script: {}".format(current_filename))
        logging.info("Session: {}".format(session_name))

        print("Create NEW table? (Y/N)")
        new_table = input()
        if "y" in new_table.lower():
            tablename, feature_type, num_features, num_feature_columns, label_column, label_data_type, noise = user_input.create_new_table(database)
            logging.info("Table {} saved in the database {} successfully.".format(tablename,database))
        elif "exit" in new_table.lower():
            raise ExitApp()
        
        paths, labels = featfun.collect_audio_and_labels()
        
        print("Would you like to extract the features and save the data to this SQL table? (Y/N)")
        cont = input()
        if "y" in cont.lower() or cont == "":
            pass
        else:
            raise ExitApp()
        
        if "tablename" not in locals():
            print("What is the SQL table to insert this data?")
            tablename = input()
            if "exit" in tablename.lower():
                raise ExitApp()
    
        logging.info("Database: {}\nTable:{}\nFeatures: {}\nNumber of Features: {}\nLabel Column: {}\nLabel Data Type: {}\nNoise: {}\n".format(database,tablename,feature_type,num_features,label_column,label_data_type,noise))
        
        limit = user_input.set_limit()
        
        print("Now extracting the features: {}".format(feature_type))
        start_feature_extraction = time.time()
        if noise:
            '''
            Need to put in functionality to include noise in training
            '''
            pass
        
        dict_data = {}
        for i, wav in enumerate(paths):
            if limit:
                if i <= limit:
                    features, extracted = featfun.get_features(wav,feature_type,num_features,num_feature_columns,noise)
                    dict_data = featfun.organize_data(dict_data,labels[i],features)
            else:
                features, extracted = featfun.get_features(wav,feature_type,num_features,num_feature_columns,noise)
                dict_data = featfun.organize_data(dict_data,labels[i],features)
        
        logging.info("Extracted features: {}".format(", ".join(extracted)))

        data_prepped = featfun.prep_data4sql(dict_data)
        end_feature_extraction = time.time()
        logging.info("Duration of feature extraction: {} minutes".format((end_feature_extraction-start_feature_extraction)/60))
        
        start_saving_data = time.time()
        logging.info("Number of samples: {}".format(len(data_prepped)))
        user_input.save2sql(database,tablename,data_prepped)
        end_saving_data = time.time()
        logging.info("Duration of saving data: {} minutes".format((end_saving_data-start_saving_data)/60))
        
    except ExitApp:
        print("Have a good day!")
        logging.info("User exited app.")
    except FeatureExtractionError as e:
        logging.exception("Error occurred in feature extraction: {}".format(e))
    except Error as e:
        logging.exception("Database error: {}".format(e))
    except Exception as e:
        logging.exception("Error occurred: {}".format(e))
    finally:
        end = time.time()
        duration = round((end-start)/60,2)
        logging.info("Duration: {} minutes".format(duration))


if __name__=="__main__":
    database = "speech_features.db"
    main(script_purpose="speech_feature_extraction",database=database)
