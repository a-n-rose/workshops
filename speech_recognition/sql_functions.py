import sqlite3
from sqlite3 import Error
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
    conn = None
    try:
        conn = sqlite3.connect(database)
        c = conn.cursor()
        
        cols = []
        for i in range(num_features):
            cols.append("'{}' REAL".format(i))
        cols_str = ", ".join(cols)
        
        msg = '''CREATE TABLE IF NOT EXISTS %s(sample_id INTEGER PRIMARY KEY, speaker_id INT, %s, speaker_sex INT) ''' % (table,cols_str)
        
        c.execute(msg)
        conn.commit()
        print("The table {} has been successfully saved.".format(table))
    except Error as e:
        print("Database error: {}".format(e))
    finally:
        if conn:
            conn.close()
            print("The database has been closed.")
            
