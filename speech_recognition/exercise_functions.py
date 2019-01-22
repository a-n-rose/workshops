'''
Script that contains functions easy to use as programming exercises.

'''


def feature_column_prep(num_features, tablename):
    '''
    If the word 'delta' is in the tablename:
    add twice the numer of feature columns to the number of features
    
    If the word 'pitch' is in the tablename:
    add just one to the number of features
    
    return that value
    '''
    columns_adjusted = num_features
    if 'delta' in tablename.lower():
        columns_adjusted += columns_adjusted*2
    if 'pitch' in tablename.lower():
        columns_adjusted +1
    return columns_adjusted
