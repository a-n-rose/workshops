import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa
import pickle
import time

import feature_extraction_functions as featfun

#variables to set:
feature_type = "fbank" # "mfcc", "stft"
num_filters = 40 # 13, None
delta = False # True
noise = True #False
sampling_rate = 16000
window = 25
shift = 10
timesteps = 5
context_window = 5
frame_width = context_window*2 + 1



#collect labels
data_path = "./data"
labels_class = featfun.collect_labels(data_path)
print(labels_class)

#create labels-encoding dictionary
labels_sorted = sorted(labels_class)
dict_labels_encoded = {}
for i, label in enumerate(labels_sorted):
    dict_labels_encoded[label] = i

#collect filenames
paths, labels_wavefile = featfun.collect_audio_and_labels(data_path)
noise_path = "./data/_background_noise_/doing_the_dishes.wav"

#to balance out the classes, find label w fewest recordings
max_num_per_class, min_label = featfun.get_min_samples_per_class(labels_class,labels_wavefile)

#create dictionary with labels and their indices in the lists: labels_wavefile and paths
#useful in separating the indices into balanced train, validation, and test datasets
dict_class_index_list = featfun.make_dict_class_index(labels_class,labels_wavefile)
            
max_nums_train_val_test = featfun.get_max_nums_train_val_test(max_num_per_class)

#randomly assign indices to train, val, test datasets:
dict_class_dataset_index_list = featfun.assign_indices_train_val_test(labels_class,dict_class_index_list,max_nums_train_val_test)

filename_save_data = "{0}_{1}_delta{2}_noise{3}_sr{4}_window{5}_shift{6}_timestep{7}_framewidth{8}".format(feature_type,num_filters,delta,noise,sampling_rate,window,shift,timesteps,frame_width)
train_val_test_filenames = []


for i in ["train","val","test"]:
    new_path = "./data_{}{}_{}/".format(feature_type,num_filters,i)
    train_val_test_filenames.append(dirname+filename_save_data)
    try:
        os.makedirs(new_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


start_feature_extraction = time.time()

for i in tqdm(range(3)):
#extract train data and save to pickel file
    dataset_index = i   # 0 = train, 1 = validation, 2 = test
    limit = int(max_nums_train_val_test[dataset_index]*.01)
    train_features = featfun.get_feats4pickle(labels_class,dict_labels_encoded,train_val_test_filenames[dataset_index],max_nums_train_val_test[dataset_index],dict_class_dataset_index_list,paths,labels_wavefile,feature_type,num_filters,timesteps,frame_width,limit=limit,delta=False,noise_wavefile=noise_path,vad=True,dataset_index=dataset_index)

end_feature_extraction = time.time()
print("Duration of feature extraction: {} minutes".format(round((end_feature_extraction-start_feature_extraction)/60,2)))

