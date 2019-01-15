'''
Try to get this script to run. It should allow you to create a database and enter new users and their ages.

Replace the #######s in the code with the following:


'''
from sql_exercise_broken_functions import ExitApp, start_section, get_username, get_age, set_up_sql_table, insert_data_sql


if __name__=="__main__":
    
    try:
        
        print("\nSQLite3 Exercise\n")
        start = start_section()
        if start == False:
            raise ExitApp("\nHave a good day!\n")    

        database = "test.db"
        tablename = "test"
        variable_list = [("id","INTEGER PRIMARY KEY"), ("username","TEXT"),("age","INT")] 
        
        set_up_sql_table(database,tablename,variable_list)
        
        
        print("\nHow many users do you want to make up?")
        num_fake_users = input()
        
        #collect data
        fake_users = []
        if num_fake_users.isdigit():
            for i in range(int(num_fake_users)):
                username = get_username()
                age = get_age(username)
                fake_users.append((username,age))
        
        #insert data
        insert_data_sql(database,tablename,fake_users)
    
    except ExitApp as e:
        print(e)
