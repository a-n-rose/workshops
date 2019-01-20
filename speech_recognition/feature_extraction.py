'''
Extracts speech features and puts into SQL database

I want to see if 

MFCC vs FBANK
dominant frequency VS fundamental frequency

work better/worse for 1) speech recognition 2) gender classification 3) handling noise
'''

import user_input 
from errors import ExitApp
import feature_extraction_functions as featfun 



if __name__=="__main__":
    
    try:
        database = "speech_commands.db"
        feature_type = "mfcc_pitch"
        num_features = 41
        noise = False
        
        print("Create NEW table? (Y/N)")
        new_table = input()
        if "y" in new_table.lower():
            feature_type, num_features, noise = user_input.create_new_table(database)
        elif "exit" in new_table.lower():
            raise ExitApp()
        
        
        paths, labels = featfun.collect_audio_and_labels()
        
        if noise:
            '''
            Need to put in functionality to include noise in training
            '''
            pass
        
        dict_data = {}
        for i, wav in enumerate(paths):
            if i <= 10:
                features = featfun.get_features(wav,feature_type,num_features,noise)
                dict_data, ex_speaker = featfun.organize_data(dict_data,labels[i],features)
                
        #need to save to sql table
        
        
    except ExitApp:
        print("Have a good day!")
