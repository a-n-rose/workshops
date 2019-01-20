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
        database = "speech_commands.db"
    if feature_type is None:
        feature_type = "mfcc_pitch"
    if num_features is None:
        num_features = 41
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
            tablename, feature_type, num_features, label_column, label_data_type, noise = user_input.create_new_table(database)
            logging.info("Table {} saved in the database {} successfully.".format(tablename,database))
        elif "exit" in new_table.lower():
            raise ExitApp()
        
        
        paths, labels = featfun.collect_audio_and_labels()
        
        print("Would you like to extract the features and save the data to an SQL table? (Y/N)")
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
        
        
        if noise:
            '''
            Need to put in functionality to include noise in training
            '''
            pass
        
        dict_data = {}
        for i, wav in enumerate(paths):
            #if i <= 100:
            features = featfun.get_features(wav,feature_type,num_features,noise)
            dict_data = featfun.organize_data(dict_data,labels[i],features)
                
        #need to save to sql table
        data_prepped = featfun.prep_data4sql(dict_data)
        logging.info("Number of samples: {}".format(len(data_prepped)))
        user_input.save2sql(database,tablename,data_prepped)
        
    except ExitApp:
        print("Have a good day!")
        logging.info("User exited app.")
    except FeatureExtractionError as e:
        logging.info("Error occurred in feature extraction: {}".format(e))
    except Error as e:
        logging.exception("Database error: {}".format(e))
    except Exception as e:
        logging.exception("Error occurred: {}".format(e))
    finally:
        end = time.time()
        duration = (end-start)/60
        logging.info("Duration: {} minutes".format(duration))


if __name__=="__main__":
    main(script_purpose="speech_feature_extraction")
