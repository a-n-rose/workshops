'''
This script serves to get the speech files prepared for training neural networks, with "matched" noise added to the training data.

Speech was collected from the Saarbücken Voice Database
'''

import numpy as np
import librosa
import sqlite3
from sqlite3 import Error
import glob
from pathlib import Path
import time
import random
import math


def collect_filenames(filename):
    filenames = []
    for wav in glob.glob("./data/{}/*.wav".format(filename)):
        filenames.append(wav)
    return filenames

def get_speaker_id(path):
    '''
    databases often save relevant data in the name of file. This function extracts the user id, i.e. the first parts of the .wav filename:
    '''
    sp = Path(path).parts[2]
    sp_id = sp.split("-")[0]
    return sp_id

def match_length(noise,sr,desired_length):
    noise2 = np.array([])
    final_noiselength = sr*desired_length
    original_noiselength = len(noise)
    frac, int_len = math.modf(final_noiselength/original_noiselength)
    for i in range(int(int_len)):
        noise2 = np.append(noise2,noise)
    if frac:
        max_index = int(original_noiselength*frac)
        end_index = len(noise) - max_index
        rand_start = random.randrange(0,end_index)
        noise2 = np.append(noise2,noise[rand_start:rand_start+max_index])
    if len(noise2) != final_noiselength:
        diff = int(final_noiselength - len(noise2))
        if diff < 0:
            noise2 = noise2[:diff]
        else:
            noise2 = np.append(noise2,np.zeros(diff,))
    return(noise2)

def scale_noise(np_array,factor):
    '''
    If you want to reduce the amplitude by half, the factor should equal 0.5
    '''
    return(np_array*factor)

def load_speech_add_noise(wavefile,noise_samples=None):
    y, sr = librosa.load(wavefile, sr=16000, res_type= 'kaiser_fast')
    # opportunity to improve data for DL training:
    # some data has a lot of silence at beginning and ending of recordings
    # can apply a voice activity detection (VAD) function to only extract features where speech is present --> reduces amount of redundant training data
    
    if noise_samples is None:
        rand_scale = 0.0
    else:
        #apply noise at various levels
        rand_scale = random.choice([0.0,0.25,0.5,0.75,1.0])
    
    if rand_scale:
        
        #apply *known* environemt noise to signal
        total_length = len(y)/sr
        envnoise_scaled = scale_noise(noise_samples,rand_scale)
        envnoise_matched = match_length(envnoise_scaled,sr,total_length)
        if len(envnoise_matched) != len(y):
            diff = int(len(y) - len(envnoise_matched))
            if diff < 0:
                envnoise_matched = envnoise_matched[:diff]
            else:
                envnoise_matched = np.append(envnoise_matched,np.zeros(diff,))
        y += envnoise_matched
        
    mfccs = librosa.feature.mfcc(y, sr, n_mfcc=40,hop_length=int(0.010*sr),n_fft=int(0.025*sr))
    
    #rows = 40
    #columns = time
    #want these to be switched:
    mfccs = np.transpose(mfccs)
    
    return mfccs


def collect_features(dict_speech_features, filename, group, noise_samples=None):
    print("now processing {} speech.".format(group))
    wavefiles = collect_filenames(filename)
    for wav in wavefiles:
        sp_id = get_speaker_id(wav)
        mfcc = load_speech_add_noise(wav,noise_samples)
        
        if group == "female":
            sex = 0
        else:
            sex = 1
        #add value attributed to sex (0 = female, 1 = male)
        
        dict_speech_features[sp_id] = (mfcc, sex)
    print("successfully extracted features")
    return dict_speech_features

def dataprep_SQL(dict_speech_features):
    ''' 
    I need to get each set of features into a tuple.
    '''
    prepped_data = []
    for key, value in dict_speech_features.items():
        # key = speaker id
        # value[0] = 40 MFCC values (each representing 25ms of speech data...)
        # value[1] = sex (0 = female, 1 = male)
        
        speaker_id = key
        sex = value[1]
        
        for row in value[0]: #get the 40 MFCC values for each segment of 25ms - there will be many!
            features = list(row)
            features.insert(0,speaker_id) #insert at index 0 the speaker ID --> corresponds to first row of SQL table
            features.append(sex) #add *at the end* the sex --> corresponds to last row of SQL table
            prepped_data.append(tuple(features)) #turn into tuple - tuples are immutable
        
    return prepped_data

def save_data_sql(prepped_data, database, table_name):
    try:
        conn = sqlite3.connect(database)
        c = conn.cursor()
        
        num_cols = len(prepped_data[0])
        
        cols = ""
        for i in range(num_cols):
            if i != num_cols-1:
                cols += " ?,"
            else:
                cols += " ?"
                
        msg = '''INSERT INTO %s VALUES(NULL, %s)''' % (table_name,cols)
        
        c.executemany(msg, prepped_data)
        conn.commit()
        
        print("All speech and noise data saved successfully!")
    except Error as e:
        print("Database Error: {}".format(e))
    finally:
        if conn:
            conn.close()
    
    return None

    
if __name__=="__main__":
    
    conn = None
    start = time.time()
    
    #need relevant infos:
    database = "male_female_speech_svd.db"
    noise = None
    if noise is not None:
        #enter your newly created noise wavefile path - if "no noise" then this will be ignored
        noise_path = "./data/background_noise.wav"
        
        noise_samples, sr = librosa.load(noise_path, sr=16000, res_type= 'kaiser_fast')
    else:
        noise_samples = None
        
    #initialize the dictionary that will collect the speech features according to speaker id
    # perk about dictionaries?
    # they don't let you enter in more than one kind of key --> you will get a key error 
    dict_speech_features = {}
    
    try:
        dict_speech_features = collect_features(dict_speech_features,"female_speech","female", noise_samples)
        dict_speech_features = collect_features(dict_speech_features,"male_speech","male",noise_samples)
        
        #prep the dictionary to insert data into SQL table
        data_prepped_4_SQL = dataprep_SQL(dict_speech_features)
        
        #insert data to SQL table
        save_data_sql(data_prepped_4_SQL, database, table_name)
        
    except KeyError as e:
        print("The speaker ID was repeated. Check for duplicates in your data.")
        
    finally:
        end = time.time()
        print("Total time: {} seconds".format(round(end - start),3))
