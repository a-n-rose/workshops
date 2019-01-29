'''
Extracts speech features and saves as spectrogram images

I want to see if 

MFCC vs FBANK vs STFT

work better/worse for 1) speech recognition 2) gender classification 3) handling noise

Do a check to ensure no ids are in the same groups... (ie. mixing of data in labels)

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


def get_max_nums_train_val_test(max_num_per_class):
    max_train = int(max_num_per_class*.8)
    max_val = int(max_num_per_class*.1)
    max_test = int(max_num_per_class*.1)
    sum_max_nums = max_train + max_val + max_test
    if max_num_per_class > sum_max_nums:
        diff = max_num_per_class - sum_max_nums
        max_train += diff
    return max_train, max_val, max_test

def get_train_val_test_indices(list_length):
    indices_ran = list(range(list_length))
    random.shuffle(indices_ran)
    train_len = int(list_length*.8)
    val_len = int(list_length*.1)
    test_len = int(list_length*.1)
    sum_indices = train_len + val_len + test_len
    if sum_indices != list_length:
        diff = list_length - sum_indices
        train_len += diff
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
    return train_indices, val_indices, test_indices


def main(script_purpose,split=False):
    current_filename = os.path.basename(__file__)
    session_name = get_date() #make sure this session has a unique identifier - link to model name and logging information
    path_to_data = "./data_3word"
    start = time.time()
    
    try:
    
        start_logging(script_purpose)
        logging.info("Running script: {}".format(current_filename))
        logging.info("Session: {}".format(session_name))

        tablename, feature_type, num_features, num_feature_columns, noise = user_input.set_variables()

        paths, labels = featfun.collect_audio_and_labels(path_to_data)
        noise_path = "./_background_noise_/doing_the_dishes.wav"
        
        label_list = [i[1] for i in labels]
        class_labels = list(set(label_list))
        print("The labels found: {}".format(class_labels))
        
        #find out number of recordings from each class
        #ask user if they want to balance out the data?
        dict_class_distribution = featfun.get_class_distribution(class_labels,label_list)
        min_val = (1000000,None)
        for key, value in dict_class_distribution.items():
            if value < min_val[0]:
                min_val = (value, key)
        print("Number of wave files for each class:\n\n{}\n\n".format(dict_class_distribution))
        print("Chosen max number of files from each class = {}".format(min_val[0]))
        print("\n\nDo you approve this? (Y/N)")
        approve = input()
        if 'exit' in approve.lower():
            raise ExitApp()
        elif 'y' in approve.lower():
            pass
        elif 'n' in approve.lower():
            print("Okay... woring on that functionality")
            raise ExitApp()
        else:
            raise ExitApp()
        
        max_num_per_class = min_val[0]

        
        #create dictionary w indices to labels and paths for each class
        dict_class_index_list = {}
        for label in class_labels:
            dict_class_index_list[label] = []
            for i, label_item in enumerate(label_list):
                if label == label_item:
                    dict_class_index_list[label].append(i)

        #get num per training/validation/test datasets
        max_nums_train_val_test = get_max_nums_train_val_test(max_num_per_class)
            
        #randomly assign indices to train, val, test datasets:
        dict_class_dataset_index_list = {}
        for label in class_labels:
            tot_indices = dict_class_index_list[label]
            tot_indices_copy = tot_indices.copy()
            random.shuffle(tot_indices_copy)
            train_indices = tot_indices_copy[:max_nums_train_val_test[0]]
            val_indices = tot_indices_copy[max_nums_train_val_test[0]:max_nums_train_val_test[0]+max_nums_train_val_test[1]]
            test_indices = tot_indices_copy[max_nums_train_val_test[0]+max_nums_train_val_test[1]:max_nums_train_val_test[0]+max_nums_train_val_test[1]+max_nums_train_val_test[2]]
            dict_class_dataset_index_list[label] = [train_indices,val_indices,test_indices]
        
        print()
        print("Name for directory to save feature images:")
        new_directory = input()
        if 'exit' in new_directory.lower():
            raise ExitApp()
        
        train_val_test_dirs = []
        for i in ["train","val","test"]:
            train_val_test_dirs.append(new_directory+"_{}".format(i))
        
        start_feature_extraction = time.time()


        for label in class_labels:
            
            for i, directory in enumerate(train_val_test_dirs):
                try:
                    os.makedirs(directory)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise

        
                dict_new_paths = {}
            
                new_path = './{}/{}/'.format(directory,label)
                try:
                    os.makedirs(new_path)
                    dict_new_paths[label] = new_path
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise


                limit = int(max_nums_train_val_test[i]*.3)
                #limit=None
                num_pics = max_nums_train_val_test[i]
                msg = "\nExtracting features from {} samples. \nImages will be saved in the directory {}".format(num_pics,new_path)
                print(msg)
                logging.info(msg)
                frame_width = 19
                time_step = 6
                logging.info("extracting features from wavefiles. Limit = {}".format(limit))
                paths_list_dataset = []
                labels_list_dataset = []
                
                train_val_test_index_list = dict_class_dataset_index_list[label]
                for k in train_val_test_index_list[i]:
                    paths_list_dataset.append(paths[k])
                    labels_list_dataset.append(label_list[k])

                        
                print("Extracting features from class: {}".format(label))
                
                for j, wav in enumerate(paths_list_dataset):
                    if limit:
                        if j <= limit:
                            featfun.save_chroma(wav,split,frame_width,time_step,feature_type,num_features,num_feature_columns,noise,dict_new_paths[labels_list_dataset[j]],noise_path,vad_noise=True)
                    else:
                        featfun.save_chroma(wav,split,frame_width,time_step,feature_type,num_features,num_feature_columns,noise,dict_new_paths[labels_list_dataset[j]],noise_path,vad_noise=True)
        

        end_feature_extraction = time.time()
        logging.info("Duration setup: {} minutes".format(round((start_feature_extraction-start)/60,2)))
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
    main(script_purpose="speech_recognition_balanced", split=False)
