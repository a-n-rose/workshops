
'''
Script outline

1) load data

2) prep data --> categorical data, one-hot-encoding, dimensionality

3) train model

4) save model

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



def main(script_purpose,database=None,tablename=None):
    current_filename = os.path.basename(__file__)
    session_name = get_date() #make sure this session has a unique identifier - link to model name and logging information
    
    #set default values
    if database is None:
        database = "speech_features.db"
        
    start = time.time()
    
    try:
    
        start_logging(script_purpose)
        logging.info("Running script: {}".format(current_filename))
        logging.info("Session: {}".format(session_name))
        
        ######################################################################
        
        #load data
        data = user_input.load_data(database,tablename)

        #prep data
        #encode categorical data
        #save labels to csv file
        features_start_stop =[2,-1]
        labels_col = -1
        X, y = featfun.encode_data(data,features_start_stop,labels_col,session_name)
        logging.info("Number of samples loaded: {}".format(len(X)))
        
        ######################################################################
        
    except ExitApp:
        print("Have a good day!")
        logging.info("User exited app.")
    except Error as e:
        logging.exception("Database error: {}".format(e))
    except Exception as e:
        logging.exception("Error occurred: {}".format(e))
    finally:
        end = time.time()
        duration = (end-start)/60
        logging.info("Duration: {} minutes".format(duration))


if __name__=="__main__":
    main(script_purpose="speech_feature_prep_train_model_speech_recognition",database="speech_commands.db",tablename="mfcc_pitch_utteranceID_41_no_noise")
