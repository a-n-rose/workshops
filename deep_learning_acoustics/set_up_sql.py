'''
Here we will set up the database to save our speech features.

It's pretty simple here. We need 1 column for each sample ID (each speaker will have many samples with MFCC features), 1 columns for the speaker ID, 40 columns for the 40 MFCC features, 2 column for the frequency values (fundamental vs dominant frequencies), and 1 column designating the sex of the speaker. This means a total of 43 columns. (note, SQL tables can handle up to 2000 columns)

'''
import sqlite3



#create table to save speech features
def create_sql_table(tablename, num_mfcc):
    #40 numbers are a lot to write out! 
    #instead of writing 0 REAL,1 REAL,2 REAL,3 REAL,4 REAL,5 REAL... blah blah:
    cols = []
    for i in range(num_mfcc):
        cols.append("'{}' REAL".format(i))
    cols_str = ", ".join(cols)
    
    msg = ''' CREATE TABLE IF NOT EXISTS %s(sample_id INTEGER PRIMARY KEY, speaker_id INT, %s, fund_freq REAL, dom_freq, speaker_sex INT)''' % (tablename,cols_str)
    
    c.execute(msg)
    conn.commit()
    return None

if __name__ == "__main__":
    try:
        #name our database
        database = "male_female_speech_svd.db"
        num_mfcc = 40
        tablename = "features_mfcc_freq"

        conn = sqlite3.connect(database)
        c = conn.cursor()
        
        create_sql_table(tablename,num_mfcc)
        
        print("Successfully created the database and table.")
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()
