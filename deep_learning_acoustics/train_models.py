'''
Script to load data and train ConvNet and LSTM models

Goal: to train a classifier to classify speech as female or male.
'''

import sqlite3
import numpy as np
import pandas as pd
import random
import math
import time

#for the models
import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Conv2D, Flatten, LSTM, MaxPooling2D, Dropout, TimeDistributed

from errors import TotalSamplesNotAlignedSpeakerSamples


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


def separate_data_train_val_test(data,ids_train, ids_val, ids_test):
    cols = data.columns
    col_id = cols[1]
    data["dataset"] = data[col_id].apply(lambda x: 0 if x in ids_train else(1 if x in ids_val else 2))
    
    train = data[data["dataset"]==0]
    val = data[data["dataset"]==1]
    test = data[data["dataset"]==2]
    
    return train, val, test

def fill_matrix_speaker_samples_zero_padded(matrix2fill, row_id, data_supplied, indices, speaker_label, len_samps_per_id, label_for_zeropadded_rows,context_window_size):
    '''
    This function fills a matrix full of zeros with the same number of rows dedicated to 
    each speaker. 
    
    If the speaker has too many samples, not all will be included. 
    If the speaker has too few samples, only the samples that will complete a full window will
    be included; the rest will be replaced with zeros/zero padded.
    
    
    1) I need the len of matrix, to be fully divisible by len_samps_per_id 
    
    2) len_samps_per_id needs to be divisible by context_window_total (i.e. context_window_size * 2 + 1)
    
    2) label column assumed to be last column of matrix2fill
    
    #mini test scenario... need to put this into unittests
    empty_matrix = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    
    #each id has 3 rows
    data_supplied = np.array([[2,2,3],[2,2,3],[2,2,3],[2,2,3],[2,2,3],[2,2,3],[4,4,5],[4,4,5],[4,4,5],[4,4,5],[4,4,5],[4,4,5],[4,4,5],[4,4,5],[4,4,5],[4,4,5],[4,4,5],[4,4,5],[4,4,5],[4,4,5],[1,1,7],[1,1,7],[1,1,7],[1,1,7],[1,1,7],[1,1,7],[1,1,7],[1,1,7],[1,1,7],[1,1,7]])
    
    indices_too_few = [0,1,2,3,4,5] #too few samples (6/10)  (total_window_size = 5) 
    
    label_too_few = 1
    
    indices_too_many = [6,7,8,9,10,11,12,13,14,15,16,17,18,19] #too many (14/10) (total_window_size = 5) 
    
    label_too_many = 0
    
    indices_just_right = [20,21,22,23,24,25,26,27,28,29] #10/10 (total_window_size = 5) 
    
    label_just_right = 1
    
    len_samps_per_id = 10
    
    label_for_zeropadded_rows = 2
    
    empty_matrices should be:
    
    row_id = 0 --> row_id = 10
    
    matrix_too_few = np.array([[2,2,3,1],[2,2,3,1],[2,2,3,1],[2,2,3,1],[2,2,3,1],[0,0,0,2],[0,0,0,2],[0,0,0,2],[0,0,0,2],[0,0,0,2],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    
    row_id = 10 --> row_id = 20
    matrix_too_many = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[4,4,5,0],[4,4,5,0],[4,4,5,0],[4,4,5,0],[4,4,5,0],[4,4,5,0],[4,4,5,0],[4,4,5,0],[4,4,5,0],[4,4,5,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    
    row_id = 20 --> row_id = 30
    matrix_too_many = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,1,7,1],[1,1,7,1],[1,1,7,1],[1,1,7,1],[1,1,7,1],[1,1,7,1],[1,1,7,1],[1,1,7,1],[1,1,7,1],[1,1,7,1]])
    '''
    try:
        frame_width = context_window_size*2+1
        
        tot_samps_speaker = len(indices)
        tot_samps_sets = tot_samps_speaker//frame_width
        tot_samps_possible = tot_samps_sets * frame_width
        
        if tot_samps_possible > len_samps_per_id:
            tot_samps_possible = len_samps_per_id
            indices = indices[:tot_samps_possible]
        
        #keep track of the samples put into the new matrix
        #don't want samples to exceed amount set by variable 'len_samps_per_id'
        samp_count = 0
        
        for index in indices:
            
            #samples only get added to matrix if fewer than max number
            if samp_count < len_samps_per_id and row_id < len(matrix2fill):
                new_row = np.append(data_supplied[index],speaker_label)
                matrix2fill[row_id] = new_row
                samp_count += 1
                row_id += 1
            else:
                if row_id >= len(matrix2fill):
                    raise TotalSamplesNotAlignedSpeakerSamples("Row id exceeds length of matrix to fill.")
            # if all user samples used, but fewer samples put in matrix than max amount, zero pad
            if samp_count < len_samps_per_id and samp_count == tot_samps_possible:
                zero_padded = len_samps_per_id - samp_count
                
                if np.modf(zero_padded/frame_width)[0] != 0.0:
                    raise TotalSamplesNotAlignedSpeakerSamples("Zero padded rows don't match window frame size")
                
                for row in range(zero_padded):
                    #leave zeros, just change label
                    matrix2fill[row_id][-1] = label_for_zeropadded_rows
                    row_id += 1
                    samp_count += 1
            
            #once all necessary samples put into matrix, leave loop and continue w next speaker 
            elif samp_count == len_samps_per_id:
                break
            
            #samp_count should not be greater than len_samps_per_id... if it is, something went wrong.
            elif samp_count > len_samps_per_id:
                raise TotalSamplesNotAlignedSpeakerSamples("More samples collected than max amount")

    except TotalSamplesNotAlignedSpeakerSamples as e:
        print(e)
        row_id = False
    
    return matrix2fill, row_id


def zero_pad_data(data,ids,context_window_size, max_num_samples):
    '''
    need to ensure each speaker has frames w sizes according to the context_window_size (i.e. context_window_size*2+1) -- the context window are the frames surrounding the sample of a particular class. If context window = 9, the whole frame would be 19.

    make every sample set (from each speaker) same number (based off of largest sample size)
    zero pad if shorter
    ignore samples if longer
    
    zero padded values given label == 2 so that real labels (i.e. 0 = healthy, 1 = clinical) are not affected.
    '''
    num_speakers = len(ids)
    frame_width = context_window_size*2+1
    samples_per_speaker_zero_padded = (max_num_samples//frame_width) * frame_width
    #len of new matrix with zero padded rows
    numrows_zeropadded_data = samples_per_speaker_zero_padded * num_speakers
    
    data_zeropadded = np.zeros((numrows_zeropadded_data,41)) #40 = num features + 1 label column

    # MONSTER FUNCTION!!
    # need to insert speech data according to user id and pad w zeros based on how many samples each speaker has available
    # need to keep track of row id and ensure it corresponds w where the row id should be
    
    
    feature_cols = list(range(2,42)) #column names of MFCC coeffecients are 2,3,4,..41
    features = data.loc[:,feature_cols].values # put all values into MATRIX
    id_label_cols = [1,42] # 1 = id, 42 = female/male 
    ids_labels = data.loc[:,id_label_cols].values
    
    # initialize row_id as 0, and it will get updated in the function
    # this helps me know that the right data is getting inserted in the right row of the new matrix
    row_id = 0
    
    # this doesn't really matter - not processed by the neural network
    # just makes me feel better that it's not labeled 0, 1 w the rest of the data
    label_for_zeropadded_rows = 2
    
    
    try:
        if np.modf(numrows_zeropadded_data/samples_per_speaker_zero_padded)[0] != 0.0:
            raise TotalSamplesNotAlignedSpeakerSamples("Length of matrix does not align with total samples for each speaker")
        
        for speaker in ids:
            equal_speaker = ids_labels[:,0] == speaker
            indices = np.where(equal_speaker)[0]
            
        
            #get label for speaker
            label = ids_labels[indices[0],1]
            
            data_zeropadded, row_id = fill_matrix_speaker_samples_zero_padded(data_zeropadded,row_id,features,indices,label,samples_per_speaker_zero_padded,label_for_zeropadded_rows,context_window_size)

    except TotalSamplesNotAlignedSpeakerSamples as e:
        print(e)
        data_zeropadded = None
    return data_zeropadded
    
def shape_data_dimensions_ConvNet_LSTM(data_zeropadded,samples_per_speaker_zero_padded, context_window_size=None):
    '''
    prep data shape for ConvNet+LSTM:
    shape = (num_speakers, num_sets_per_speaker; num_frames_per_set; num_features_per_frame; grayscale)
    
    If ConvNet and LSTM put together --> (66,32,19,120,1) if 66 speakers
    - ConvNet needs grayscale 
    - LSTM needs num_sets_per_speaker 
    
    If separate:
    - Convent needs grayscale (19,120,1)
    - LSTM needs number features in a series, i.e. 19 (19,120)
    '''
    
    if context_window_size is None:
        context_window_size = 9
    
    #separate features from labels:
    features = data_zeropadded[:,:-1]
    labels = data_zeropadded[:,-1]
    num_frame_sets = samples_per_speaker_zero_padded//(context_window_size*2+1)
    
    num_sets_samples = len(data_zeropadded)//num_frame_sets
    
    num_speakers = len(data_zeropadded)//samples_per_speaker_zero_padded
    
    #make sure only number of samples are included to make up complete context window frames of e.g. 19 frames (if context window frame == 9, 9 before and 9 after a central frame, so 9 * 2 + 1)
    check = len(data_zeropadded)//num_frame_sets
    if math.modf(check)[0] != 0.0:
        print("Extra Samples not properly removed")
    else:
        print("No extra samples found")
    
    #reshaping data to suit ConvNet + LSTM model training. 
    #see notes at top of function definition
    X = features.reshape(len(data_zeropadded)//samples_per_speaker_zero_padded,samples_per_speaker_zero_padded//(context_window_size*2+1),context_window_size*2+1,features.shape[1],1)
    y_indices = list(range(0,len(labels),samples_per_speaker_zero_padded))
    y = labels[y_indices]
    return X, y

if __name__=="__main__":
    
    start = time.time()
    
    database = "male_female_speech_svd.db"
    table_name = "mfcc"
    context_window_size = 9 # 9*2+1 = 19 total frame width
    frame_width = context_window_size*2+1
    
    try:
        conn = sqlite3.connect(database)
        c = conn.cursor()
        
        # get speech data
        data = get_speech_data(table_name)
    except Error as e:
        print("Database error: {}".format(e))
    finally:
        if conn:
            conn.close()
    
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
    samples_per_speaker_zero_padded = (max_samples//frame_width)*frame_width
    
    # don't want speakers to be mixed in both training and validation/test datasets:
    # normally could use sklearn's train_test_split()
    # Write own function that splits randomly, based on speaker!!
    ids_train, ids_val, ids_test = split_train_val_test(ids)
    
    #not necessary - just a little check-in
    num_train_speakers = len(ids_train)
    print("Number of speakers in training set: {}".format(num_train_speakers))
    num_val_speakers = len(ids_val)
    print("Number of speakers in validation set: {}".format(num_val_speakers))
    num_test_speakers = len(ids_test)
    print("Number of speakers in test set: {}".format(num_test_speakers))
    
    train, val, test = separate_data_train_val_test(data,ids_train, ids_val, ids_test)
    
    print(train.head())
    
    train_zeropadded =  zero_pad_data(train,ids_train,context_window_size, max_samples)
    val_zeropadded =  zero_pad_data(val,ids_val,context_window_size, max_samples)
    test_zeropadded =  zero_pad_data(test,ids_test,context_window_size, max_samples)
    
    print(train_zeropadded[:200])
    print()
    print(train_zeropadded[-200:])
    print()

    
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
    print("The dimension of this data, to be fed to both ConvNet and LSTM layers, needs to be... ")
    print("\nAll Data:\n")
    print("({},{},{},{},{})".format(A,B,C,D,E))
    
    A_train = len(ids_train)
    print("\nTraining Data:\n")
    print("({},{},{},{},{})".format(A_train,B,C,D,E))
    
    A_val = len(ids_val)
    print("\nValidation Data:\n")
    print("({},{},{},{},{})".format(A_val,B,C,D,E))
    
    A_test = len(ids_test)
    print("\nTest Data:\n")
    print("({},{},{},{},{})".format(A_test,B,C,D,E))
    
    X_train, y_train = shape_data_dimensions_ConvNet_LSTM(train_zeropadded,samples_per_speaker_zero_padded, context_window_size=context_window_size)
    
    X_val, y_val = shape_data_dimensions_ConvNet_LSTM(val_zeropadded,samples_per_speaker_zero_padded, context_window_size=context_window_size)
    
    X_test, y_test = shape_data_dimensions_ConvNet_LSTM(test_zeropadded,samples_per_speaker_zero_padded, context_window_size=context_window_size)
    
    print(X_train[:100])
    print()
    print(y_train[:100])
    
    
    #train the models!
    
    try:
        
        #TIME-FREQUENCY CONVNET
        tfcnn = Sequential()
        # feature maps = 40
        # 8x4 time-frequency filter (goes along both time and frequency axes)
        num_features = 40
        color_scale = 1
        input_size = (frame_width,num_features,color_scale)
        tfcnn.add(Conv2D(40, kernel_size=(8,4), activation='relu'))
        #non-overlapping pool_size 3x3
        tfcnn.add(MaxPooling2D(pool_size=(3,3)))
        tfcnn.add(Dropout(0.25))
        tfcnn.add(Flatten())
        
        #prepare LSTM
        tfcnn_lstm = Sequential()
        timestep = samples_per_speaker_zero_padded//frame_width
        tfcnn_lstm.add(TimeDistributed(tfcnn,input_shape=(timestep,frame_width,num_features,color_scale)))
        tfcnn_lstm.add(LSTM(timestep)) #num timesteps
        tfcnn_lstm.add(Dense(1,activation="sigmoid"))
        
        
        print(tfcnn_lstm.summary())
        
        
        #compile model
        tfcnn_lstm.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
        #train model
        tfcnn_lstm.fit(X_train, y_train, validation_data=(X_val,y_val),epochs=100)
        
        #predict test data
        pred = tfcnn_lstm.predict(X_test)
        pred = pred >0.5
        pred = pred.astype(float)
        
        #see how many were correct
        correct = 0
        for i, item in enumerate(y_test):
            if item == pred[i]:
                correct += 1
        score = round(correct/float(len(y_test)) * 100, 2)
        print("\n\nmodel earned a score of {}%  for the test data.\n\n".format(score))
        
    except Exception as e:
        print(e)
        
    finally:
        end = time.time()
        print("Total duration: {} sec.".format(round(end-start,3)))
