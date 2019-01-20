import sqlite3
import librosa
import numpy as np


def create_table(database,table,num_features):
    '''
    Need to:
    ~ connect with SQL database
    ~ create the table
    ~ create the columns w data type they will contain
    ~ commit the new table
    '''
