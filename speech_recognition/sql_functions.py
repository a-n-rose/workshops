import sqlite3
from sqlite3 import Error
import librosa
import numpy as np


def create_table(database,table,num_features,label_column,label_data_type):
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
        
        label_info = label_column+" "+label_data_type
        
        msg = '''CREATE TABLE IF NOT EXISTS %s(sample_id INTEGER PRIMARY KEY, speaker_id TEXT, utterance_id INT, %s, %s) ''' % (table,cols_str,label_info)
        print(msg)
        
        c.execute(msg)
        conn.commit()
        print("The table {} has been successfully saved.".format(table))
    except Error as e:
        print("Database error: {}".format(e))
    finally:
        if conn:
            conn.close()
            print("The database has been closed.")
    return None
            
            
def insert_data(database,table,data):
    conn = None
    try:
        conn = sqlite3.connect(database)
        c = conn.cursor()
        num_cols = len(data[0])
        print("number of columns: ",num_cols)
        placeholders = ""
        for i in range(num_cols):
            if i == num_cols-1:
                placeholders += "?"
            else:
                placeholders += "?, "
        
        msg = '''INSERT INTO %s VALUES(NULL,%s) ''' % (table,placeholders)
        
        c.executemany(msg,data)
        conn.commit()
        print("The table {} has been successfully saved.".format(table))
    except Error as e:
        print("Database error: {}".format(e))
    finally:
        if conn:
            conn.close()
            print("The database has been closed.")
    return None


def select_data(database,table,limit=None):
    conn = None
    try:
        conn = sqlite3.connect(database)
        c = conn.cursor()
        
        if limit is not None:
            table+=" LIMIT "+limit
        msg = '''SELECT * FROM %s''' % table
        
        c.execute(msg)
        data = c.fetchall()
        print("Data has been loaded from the table {}".format(table))
        return data
    
    except Error as e:
        print("Database error: {}".format(e))
    finally:
        if conn:
            conn.close()
            print("The database has been closed.")
    return None
            
