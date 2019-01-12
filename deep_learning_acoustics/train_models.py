'''
Script to load data and train ConvNet and LSTM models

Goal: to train a classifier to classify speech as female or male.
'''

import sqlite3
import numpy as np
import pandas as pd
import random
import math

##for the models
#import keras
#from keras.models import Sequential
#from keras.utils import np_utils
#from keras.layers import Dense, Conv2D, Flatten, LSTM, MaxPooling2D, Dropout, TimeDistributed



def get_speech_data(table_name):
    '''
    I use '*' below due to high number of features: over 40 columns, with speaker_id, sex, and mfcc features
    '''
    msg = ''' SELECT * FROM %s ''' % table_name
    c.execute(msg)
    data = pd.DataFrame(c.fetchall())
    return data

def randomly_choose_ids(ids,num):
    random.shuffle(ids)
    ids = ids[:num]
    return ids    

def balance_data(data):
    cols = data.columns
    col_sex = cols[-1]
    data_f = data[data[col_sex]==0]
    ids_f = get_speaker_ids(data_f)
    data_m = data[data[col_sex]==1]
    ids_m = get_speaker_ids(data_m)
    max_num_speakers = min(len(ids_f),len(ids_m))
    
    #randomly choose which female and male speakers are included:
    ids_f = randomly_choose_ids(ids_f, max_num_speakers)
    ids_m = randomly_choose_ids(ids_m, max_num_speakers)
    
    ids = list(ids_f) + list(ids_m)
    
    col_id = cols[1]
    data["selected"] = data[col_id].apply(lambda x: True if x in ids else False)
    data = data[data["selected"]==True]
    
    data = data.drop(columns="selected")
    
    return data

def get_speaker_ids(data):
    cols = data.columns
    col_id = cols[1]
    ids = data[col_id]
    ids = ids.unique()
    return(ids)

def get_num_samples_per_speaker(ids,data):
    samples_list = []
    cols = data.columns
    col_id = cols[1]
    for speaker in ids:
        samples_list.append(sum(data[col_id]==speaker))
    return samples_list


def split_train_val_test(speaker_ids,perc_train=None,perc_val=None,perc_test=None):
    '''
    Splits speakers into training, validation, and test
    default: 80-10-10 ratio
    
    should put in 'random seed' functionality..
    '''
    if perc_train is None:
        perc_train = 0.8
        perc_val = 0.1
        perc_test = 0.1
        
    num_speakers = len(speaker_ids)
    num_train = int(num_speakers * perc_train)
    num_val = int(num_speakers * perc_val)
    num_test = int(num_speakers * perc_test)
    
    train = [0] * num_train
    val = [1] * num_val
    test = [2] * num_test
    
    randomly_assigned_conditions = np.concatenate((train,val,test))
    random.shuffle(randomly_assigned_conditions)
    
    train_speakers = []
    val_speakers = []
    test_speakers = []
    
    #the number of assigned conditions might be slightly less than num_speakers
    #using int() above does not round up, only down
    if len(randomly_assigned_conditions) < num_speakers:
        diff = num_speakers - len(randomly_assigned_conditions)
        for j in range(diff):
            rand_choice = np.random.choice([0,1,2],p=[0.8,0.1,0.1])
            randomly_assigned_conditions=np.append(randomly_assigned_conditions,rand_choice) 
            
    for i in range(num_speakers):
        if randomly_assigned_conditions[i] == 0:
            train_speakers.append(speaker_ids[i])
        elif randomly_assigned_conditions[i] == 1:
            val_speakers.append(speaker_ids[i])
        elif randomly_assigned_conditions[i] == 2:
            test_speakers.append(speaker_ids[i])
            
    return train_speakers, val_speakers, test_speakers


def separate_data_train_val_test(data,train_ids, val_ids, test_ids):
    cols = data.columns
    col_id = cols[1]
    data["dataset"] = data[col_id].apply(lambda x: 0 if x in train_ids else(1 if x in val_ids else 2))
    
    train = data[data["dataset"]==0]
    val = data[data["dataset"]==1]
    test = data[data["dataset"]==2]
    
    return train, val, test


def prepare_data_dimensions_ConvNet_LSTM(data,col_name,num_speakers,context_window_size, max_num_samples):
    '''
    need to ensure each speaker has frames w sizes according to the context_window_size (i.e. context_window_size*2+1) -- the context window are the frames surrounding the sample of a particular class. If context window = 9, the whole frame would be 19.

    make every sample set (from each speaker) same number (based off of largest sample size)
    zero pad if shorter
    ignore samples if longer
    
    zero padded values given label == 2 so that real labels (i.e. 0 = healthy, 1 = clinical) are not affected.
    '''
    
    #set len of new matrix
    frame_width = context_window_size * 2 + 1
    num_samples_per_speaker = (max_num_samples//frame_width) * frame_width
    
    #len of new matrix with zero padded rows
    numrows_zeropadded_data = num_samples_per_speaker * num_speakers
    
    #create empty matrix to insert values into (more efficient this way)
    #start with zeros as I will zero pad
    data_zeropadded = np.zeros((numrows_zeropadded_data,41)) #40 = num features + 1 label column
    
    '''
    TO DO:
    
    Fill in the new matrix 
    '''
    pass
    

if __name__=="__main__":
    
    database = "male_female_speech_svd.db"
    table_name = "mfcc"
    
    conn = sqlite3.connect(database)
    c = conn.cursor()
    
    # get speech data
    data = get_speech_data(table_name)
    print("original data")
    print(data.shape)
    
    # even out data representation
    data = balance_data(data)
    print("balanced data")
    print(data.shape)
    
    # get speaker ids 
    ids = get_speaker_ids(data)
    num_speakers = len(ids)
    # get number of mfcc rows per speaker --> find out the max number of samples
    tot_user_samples = get_num_samples_per_speaker(ids,data)
    # max num samples
    max_samples = max(tot_user_samples)
    print("max num samples: {}".format(max_samples))
    
    # don't want speakers to be mixed in both training and validation/test datasets:
    # normally could use sklearn's train_test_split()
    # Write own function that splits randomly, based on speaker!!
    train_ids, val_ids, test_ids = split_train_val_test(ids)
    
    #not necessary - just a little check-in
    num_train_speakers = len(train_ids)
    print("Number of speakers in training set: {}".format(num_train_speakers))
    num_val_speakers = len(val_ids)
    print("Number of speakers in validation set: {}".format(num_val_speakers))
    num_test_speakers = len(test_ids)
    print("Number of speakers in test set: {}".format(num_test_speakers))
    
    train, val, test = separate_data_train_val_test(data,train_ids, val_ids, test_ids)
    
    
    '''
    now that we have the data separated, we need to prepare it for the models
    normally one would normalize the data by subtracting the mean and dividing by standard deviation
    with MFCCs and deep learning, it is debatable whether to do this
    so we won't for now
    
    We need to get the data into the right dimensions!!! 
    
    NUMPY TO THE RESCUE!
    
    some research uses window frames of 19, so I will feed the ConvNet "images"
    with dimensions of (19,40,1) --> 19 width of frames, with 40 features (40 MFCCs), 1 == grayscale
    to capture the time-series data, feed data into LSTM
    LSTMs need a set number to expect for each time series, which is where the "max_samples" comes in. I will feed the max number of samples (divisable by 19)
    and for speakers with fewer data samples: I will zero-pad the rest
    
    In the end, the data needs to be in the following dimension:
    (A, B, C, D)
    A = number speakers 
    B = number of samples (that have full frames w width of 19) 
    C = number of samples per frame(19) 
    D = number of features (i.e. MFCCs = 40)
    E = the color-scale (gray = 1,  color = 3 (RGB))
    
    '''
    A = num_speakers
    B = (max_samples//19) * 19 #using floor division
    C = 19 #number of samples per "picture"
    D = 40 #number of MFCC features
    E = 1 #grayscale
    print("The dimension of this data, to be fed to both ConvNet and LSTM layers, needs to be: ")
    print("({},{},{},{},{})".format(A,B,C,D,E))
