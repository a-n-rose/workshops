'''
Extracts speech features and puts into SQL database
'''

import feature_extraction_functions as fefun
from errors import ExitApp

if __name__=="__main__":
    
    try:
        
        database = "speech_commands.db"
        noise = False
        if noise:
            desc = "w_noise"
        else:
            desc = "no_noise"
        feature = "fbank" # other options: "mfcc"
        num_features = 40
        
        #set table name:
        table = "{}_{}_{}".format(feature,num_features,desc)
        
        print("The table:\n\n{}\n\n will be created in the database:\n\n{}\n\n".format(table,database))
        print("Press ENTER to continue.")
        cont = input()
        if cont != "":
            raise ExitApp("Have a good day!")
        
        fefun.create_table(database,table,num_features)
        
    except ExitApp as e:
        print(e)
