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
import random

import user_input 
from errors import ExitApp, FeatureExtractionError
import feature_extraction_functions as featfun 

import logging
from my_logger import start_logging, get_date
logger = logging.getLogger(__name__)


def get_train_val_test_indices(list_length):
    indices_ran = list(range(list_length))
    random.shuffle(indices_ran)
    train_len = int(list_length*.8)
    print(train_len)
    val_len = int(list_length*.1)
    print(val_len)
    test_len = int(list_length*.1)
    print(test_len)
    sum_indices = train_len + val_len + test_len
    if sum_indices != list_length:
        print(sum_indices)
        print(list_length)
        diff = list_length - sum_indices
        train_len += diff
    print(train_len)
    train_indices = []
    val_indices = []
    test_indices = []
    for i, item in enumerate(indices_ran):
        if i < train_len:
            train_indices.append(item)
        elif i >= train_len and i < train_len+val_len:
            val_indices.append(item)
        elif i >= train_len + val_len and i < list_length:
            test_indices.append(item)
    print(train_indices)
    print(val_indices)
    print(test_indices)
    return train_indices, val_indices, test_indices


def main(script_purpose,split=False):
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
        
        
        #set up train, validation, and test paths:
        train_val_test_indices = get_train_val_test_indices(len(paths))

        
        print("Name for directory to save feature images:")
        new_directory = input()
        
        train_val_test_dirs = []
        for i in ["train","val","test"]:
            train_val_test_dirs.append(new_directory+"_{}".format(i))
        
        start_feature_extraction = time.time()




        for i, directory in enumerate(train_val_test_dirs):
            try:
                os.makedirs(directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        
            dict_new_paths = {}
            for label in class_labels:
                new_path = './{}/{}/'.format(directory,label)
                try:
                    os.makedirs(new_path)
                    dict_new_paths[label] = new_path
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise

            
            limit = int(len(train_val_test_indices[i])*.1)
            #limit=None
            num_pics = len(train_val_test_indices[i])
            print(num_pics)
            print("LIMIT = {}".format(limit))
            frame_width = 19
            time_step = 5
            logging.info("extracting features from wavefiles. Limit = {}".format(limit))
            paths_list_dataset = []
            labels_list_dataset = []
            for k in train_val_test_indices[i]:
                paths_list_dataset.append(paths[k])
                labels_list_dataset.append(label_list[k])
                
            for j, wav in enumerate(paths_list_dataset):
                if limit:
                    if j <= limit:
                        featfun.save_chroma(wav,split,frame_width,time_step,feature_type,num_features,num_feature_columns,noise,dict_new_paths[labels_list_dataset[j]])
                else:
                    featfun.save_chroma(wav,split,frame_width,time_step,feature_type,num_features,num_feature_columns,noise,dict_new_paths[labels_list_dataset[j]])
        

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
        logging.info("Duration setup: {} minutes".format(round((start_feature_extraction-start)/60,2)))

if __name__=="__main__":
    main(script_purpose="speech_features_save_as_PNG", split=True)
