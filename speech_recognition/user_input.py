from sql_functions import create_table

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
    print("How many feature columns do you need?")
    num_features = input()
    if num_features.isdigit():
        pass
    elif "exit" in num_features.lower():
        raise ExitApp()
    else:
        print("Please enter an integer")
        num_features = get_num_features()
    return int(num_features)

def feature_type():
    print("What kind of features will you extract? (e.g. mfcc, fbank)")
    features = input()
    if "exit" in features.lower():
        raise ExitApp()
    return features

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
    features = feature_type()
    num_features = get_num_features()
    noise = add_noise()

    if noise:
        description = "w_noise"
    else:
        description = "no_noise"
    
    #set table name:
    table = "{}_{}_{}".format(features,num_features,description)
    
    print("\n\nTHE TABLE ~   {}   ~ WILL BE CREATED IN THE DATABASE ~   {}   ~".format(table,database))

    if go():
        create_table(database,table,num_features)
    return None
