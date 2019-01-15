'''

Replace the #######s in the code with the following:

Use 4 times:
input()


Use 1 time each:
lower
format
isdigit
connect
enumerate
conn.commit
executemany
execute
conn.close

'''

#for handling errors/exceptions:
class Error(Exception):
    """Base class for other exceptions"""
    pass

class ExitApp(Error):
    pass



# User input: are you ready?
def start_section():
    print("\nReady to test the script? (Y/N)")
    start = #####
    if "y" in start.lower():
        return True
    return False


# Collect User Information and working with STRINGS
def get_username():
    print("\nEnter username: ")
    username = #####
    print("\nYou have entered {}".format(username))
    print("\nIs this correct? (Y/N)")
    
    #simple way to see if the user typed "Y" or "y" or "YES" or "yes" 
    correct = #####
    if "y" in correct.#####():
        pass
    else:
        username = get_username()
    return username

def get_age(username):
    print("\nHow old is {}?".#####(username))
    age = #####
    
    #make sure the user put in an actual number:
    if age.#####():
        pass
    else:
        print("\nPlease enter an integer.")
        age = get_age(username)
    return age

def set_up_sql_table(database, tablename, variable_list):
    '''
    variable list: list of pairs indicating column name and data type
    id, INTEGER PRIMARY KEY
    username, TEXT 
    speaker_age, INT
    '''
    import sqlite3
    from sqlite3 import Error
    conn = None
    
    try:
        #connect with the database
        conn = sqlite3.######(database)
        c = conn.cursor()
    
        variable_string = ""
        for i, pair in #######(variable_list):
            if i < len(variable_list)-1:
                variable_string+=pair[0]+" "+pair[1]+", "
            else:
                variable_string+=pair[0]+" "+pair[1]
        
        msg = ''' CREATE TABLE IF NOT EXISTS %s( %s ) ''' % (tablename, variable_string)
        c.execute(msg)
        #########()
    
    except Error as e:
        print("\nDatabase error: {}\n".format(e))
    
    finally:
        if conn:
            conn.close()
    
    return None

def insert_data_sql(database, tablename, data):
    '''
    data must be a list of tuples
    '''
    import sqlite3
    from sqlite3 import Error
    conn = None
    
    try:
        conn = sqlite3.connect(database)
        c = conn.cursor()
    
        place_holders = ""
        for i in range(len(data[0])):
            if i < len(data[0])-1:
                place_holders += "?, "
            else:
                place_holders += "?"
                
        msg = '''INSERT INTO %s VALUES(NULL, %s) ''' % (tablename,place_holders)
        
        if len(data) > 1:
            c.########(msg,data)
        else:
            c.######(msg,data[0])
        conn.commit()
        print("\n\nNames and ages saved successfully!\n\n")
    except Error as e:
        print("\nDatabase error: {}\n".format(e))
    finally:
        if conn:
            #########()
        
    return None



    
