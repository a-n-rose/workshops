'''
Extracts speech features and puts into SQL database

I want to see if 

MFCC vs FBANK
dominant frequency VS fundamental frequency

work better/worse for 1) speech recognition 2) gender classification 3) handling noise
'''
import time
import os, errno
import pandas as pd

import user_input 
from errors import ExitApp, FeatureExtractionError
import feature_extraction_functions as featfun 

import logging
from my_logger import start_logging, get_date
logger = logging.getLogger(__name__)


def main(script_purpose):
    current_filename = os.path.basename(__file__)
    session_name = get_date() #make sure this session has a unique identifier - link to model name and logging information
    
    start = time.time()
    
    try:
    
        start_logging(script_purpose)
        logging.info("Running script: {}".format(current_filename))
        logging.info("Session: {}".format(session_name))

        tablename, feature_type, num_features, num_feature_columns, noise = user_input.set_variables()

        paths, labels = featfun.collect_audio_and_labels()
        
        label_list = [i[1] for i in labels]
        class_labels = list(set(label_list))
        print(class_labels)
        
        print("Name for directory to save feature images:")
        new_directory = input()
        

        try:
            os.makedirs(new_directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        dict_new_paths = {}
        for label in class_labels:
            new_path = './{}/{}/'.format(new_directory,label)
            try:
                os.makedirs(new_path)
                dict_new_paths[label] = new_path
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        start_feature_extraction = time.time()
        limit = 500
        frame_width = 19
        time_step = 5
        logging.info("extracting features from wavefiles. Limit = {}".format(limit))
        for i, wav in enumerate(paths):
            if limit:
                if i <= limit:
                    featfun.save_chroma(wav,frame_width,time_step,feature_type,num_features,num_feature_columns,noise,dict_new_paths[label_list[i]])
        

        end_feature_extraction = time.time()
        print("Duration of feature extraction: {} minutes".format(round((end_feature_extraction-start_feature_extraction)/60,2)))
    except ExitApp:
        print("Have a good day!")
        logging.info("User exited app.")
    except FeatureExtractionError as e:
        logging.exception("Error occurred in feature extraction: {}".format(e))
    except Exception as e:
        logging.exception("Error occurred: {}".format(e))
    finally:
        end = time.time()
        duration = round((end-start)/60,2)
        logging.info("Duration: {} minutes".format(duration))


if __name__=="__main__":
    main(script_purpose="speech_features_save_as_PNG")
