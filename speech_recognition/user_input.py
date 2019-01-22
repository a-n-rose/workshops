from sql_functions import create_table, insert_data, select_data
from errors import ExitApp

def add_noise():
    print("Will you add noise? (Y/N)")
    noise = input()
    if "y" in noise.lower():
        noise = True
    elif "n" in noise.lower():
        noise = False
    elif "exit" in noise.lower():
        raise ExitApp()
    return noise

def get_num_features():
    print("How many feature columns do you need? (If you want the derivatives, DON'T INCLUDE THE COLUMNS FOR THOSE VALUSE HERE)")
    num_features = input()
    if num_features.isdigit():
        pass
    elif "exit" in num_features.lower():
        raise ExitApp()
    else:
        print("Please enter an integer")
        num_features = get_num_features()
    return int(num_features)

def get_feature_type():
    print("What kind of features will you extract? (e.g. mfcc, fbank, fbank with deltas, mfcc with pitch values)")
    features = input()
    if "exit" in features.lower():
        raise ExitApp()
    return features

def get_label_column():
    print("What is the column name for the label? (e.g. word, gender, class)")
    label_column = input()
    if "exit" in label_column.lower():
        raise ExitApp()
    print("You entered {}. Is that correct? (Y/N)".format(label_column))
    correct = input()
    if "y" in correct.lower():
        pass
    elif "exit" in correct.lower():
        raise ExitApp()
    else:
        label_column = get_label_column()
    return label_column

def get_label_data_type():
    print("What type of data will the label be saved in the table? (i.e. 'INT', or 'TEXT')")
    data_type = input()
    type_options = ["INT","INTEGER","TEXT"]
    if data_type.upper() not in type_options:
        print("Please enter one of the following: {}".format(", ").join(type_options))
        data_type = label_data_type()
    elif "exit" in data_type.lower():
        raise ExitApp()
    return data_type

def go():
    print("Press ENTER to continue.")
    cont = input()
    if "exit" in cont.lower():
        raise ExitApp("Have a good day!")
    elif cont != "":
        print("Type 'exit' to end the program.")
        cont = go()
    return True

def create_new_table(database):
    feature_type = get_feature_type()
    num_features = get_num_features()
    label_column = get_label_column()
    label_data_type = get_label_data_type()
    noise = add_noise()
    
    num_features = int(num_features)
    num_feature_columns = num_features
    features_included = []
    if "mfcc" in feature_type.lower():
        features_included.append("mfcc")
    elif "fbank" or "mel filter bank" in feature_type.lower():
        features_included.append("fbank")
    if "pitch" in feature_type.lower():
        features_included.append("pitch")
        num_feature_columns += 1
    if "derivative" or "delta" or "rate of change" in feature_type.lower():
        features_included.append("delta")
        #add delta and delta-delta values
        num_feature_columns += num_features * 2 

    if noise:
        description = "w_noise"
    else:
        description = "no_noise"
    
    #set table name:
    features_standardized = "_".join(features_included)

    table = "{}_{}_{}".format(features_standardized,num_feature_columns,description)
    
    print("\n\nTHE TABLE ~   {}   ~ WILL BE CREATED IN THE DATABASE ~   {}   ~".format(table,database))

    if go():
        create_table(database,table,num_feature_columns,label_column, label_data_type)
        
    return table, features_standardized, num_features, num_feature_columns, label_column, label_data_type, noise

def save2sql(database,tablename,data_prepped):
    #print("Press ENTER to save the data to the table ~  {}  ~ in the database ~  {}  ~".format(tablename,database))
    #cont = input()
    #if cont == "":
    saved = insert_data(database,tablename,data_prepped)
    #elif "exit" in cont.lower():
        #raise ExitApp()
    #else:
        #save2sql(database,tablename,data_prepped)
    return None
    
def load_data(database,table,columns=None):
    if columns is None:
        columns = "all"
    print("Loading data from {} columns from {} in database: {}".format(columns,table,database))
    limit = set_limit()
    print("Press ENTER to continue")
    cont = input()
    if "exit" in cont.lower():
        raise ExitApp()
    elif cont.lower() == "":
        pass
    else:
        cont = load_data(database,table)
    data = select_data(database,table,limit)
    return data
    
def set_limit():
    print("Is there a limit for the data? If YES: enter an INTEGER.")
    limit = input()
    if "exit" in limit.lower():
        raise ExitApp()
    elif limit.isdigit():
        limit = int(limit)
    else:
        limit = None
    return limit
 
def get_desired_features():
    print("With which features do you want to train the model? (e.g. MFCC, FBANK, MFCC and pitch, FBANK and deltas)")
    features = input()
    if "exit" in features.lower():
        raise ExitApp()
    print("\nYou entered: \n\n{}\n\nIs this correct? (Y/N)".format(features))
    correct = input()
    if "exit" in correct.lower():
        raise ExitApp()
    elif "y" not in correct.lower():
        features = get_desired_features()
        
    return features
