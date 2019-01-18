'''
This script serves to get the speech files prepared for training neural networks, with "matched" noise added to the training data.

Speech was collected from the Saarbücken Voice Database
'''

import pandas as pd
import numpy as np
import librosa
import sqlite3
from sqlite3 import Error
import glob
from pathlib import Path
import time
import random
import math

from get_speech_features import get_samps, get_mfcc, get_fundfreq, get_domfreq


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


def collect_features(dict_speech_features, filename, group):
    print("now processing {} speech.".format(group))
    wavefiles = collect_filenames(filename)
    for wav in wavefiles:
        sp_id = get_speaker_id(wav)
        sr = 16000
        y = get_samps(wav,sr)
        mfcc = get_mfcc(y,sr)
        fundfreq = np.array(get_fundfreq(y,sr))
        fundfreq = fundfreq.reshape(len(fundfreq),1)
        
        domfreq = np.array(get_domfreq(y,sr))
        domfreq = domfreq.reshape(len(domfreq),1)
        
        features = np.concatenate((mfcc,fundfreq,domfreq),axis=1)
        
        if group == "female":
            sex = 0
        else:
            sex = 1
        #add value attributed to sex (0 = female, 1 = male)
        
        dict_speech_features[sp_id] = (features, sex)
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

    #initialize the dictionary that will collect the speech features according to speaker id
    # perk about dictionaries?
    # they don't let you enter in more than one kind of key --> you will get a key error 
    dict_speech_features = {}
    
    try:
        dict_speech_features = collect_features(dict_speech_features,"female_speech","female")
        dict_speech_features = collect_features(dict_speech_features,"male_speech","male")
        
        #prep the dictionary to insert data into SQL table
        data_prepped_4_SQL = dataprep_SQL(dict_speech_features)
        
        #insert data to SQL table
        #need relevant infos:
        database = "male_female_speech_svd.db"
        table_name = "features_mfcc_freq"
        save_data_sql(data_prepped_4_SQL, database, table_name)
        
    except KeyError as e:
        print("The speaker ID was repeated. Check for duplicates in your data.")
        
    finally:
        end = time.time()
        print("Total time: {} seconds".format(round(end - start),3))
