'''
Extracts speech features and puts into SQL database
'''

import user_input 
from errors import ExitApp



if __name__=="__main__":
    
    try:
        database = "speech_commands.db"
        
        print("Create NEW table? (Y/N)")
        new_table = input()
        if "y" in new_table.lower():
            user_input.create_new_table(database)
        else:
            raise ExitApp()
        
    except ExitApp:
        print("Have a good day!")
