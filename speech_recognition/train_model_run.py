
'''
Script outline

1) load data

2) prep data --> zeropad, encode categorical data, dimensionality

3) train model

4) save model

'''

import time
import os
from sqlite3 import Error

#for training
from sklearn.model_selection import train_test_split
#for the models
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, LSTM, MaxPooling2D, Dropout, TimeDistributed

import user_input 
from errors import ExitApp, FeatureExtractionError
import feature_extraction_functions as featfun 

import logging
from my_logger import start_logging, get_date
logger = logging.getLogger(__name__)

from exercise_functions import feature_column_prep


def main(script_purpose,database=None,tablename=None):
    current_filename = os.path.basename(__file__)
    session_name = get_date() #make sure this session has a unique identifier - link to model name and logging information
    
    #set default values
    if database is None:
        database = "speech_features.db"
        
    start = time.time()
    
    try:
    
        start_logging(script_purpose)
        logging.info("Running script: {}".format(current_filename))
        logging.info("Session: {}".format(session_name))
        
        ######################################################################
        
        #load data
        data = user_input.load_data(database,tablename)

        #necessary variables:
        id_col_index = 2 #index 0 --> sample ID, index 1 --> speaker ID
        features_start_stop_index = [3,-1]
        label_col_index = [-1]
        num_features = 40
        context_window_size = 9
        frame_width = context_window_size*2+1
        
        #add feature columns based on which features are to be expected
        num_feature_columns = feature_column_prep(num_features,tablename)
        
        #prep data
        #1) make sure each utterance has same number of samples;
        #if not, zeropad them so each has same number of samples
        data_zeropadded, samples_per_utterance, num_utterances, labels_present = featfun.prep_data(data,id_col_index,features_start_stop_index,label_col_index,num_feature_columns,frame_width,session_name)
        
        logging.info("Fixed number of samples per utterance: {}".format(samples_per_utterance))
        logging.info("Number of utterances in data: {}".format(num_utterances))
        
        X, y = featfun.shape_data_dimensions_CNN_LSTM(data_zeropadded,samples_per_utterance,frame_width)
        
        logging.info("Shape of feature data: {}".format(X.shape))
        logging.info("Shape of label data: {}".format(y.shape))
        
        #separate X and y --> training and test datasets
        
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.1)
        
        ######################################################################
        
    #train the models!
    
        #TIME-FREQUENCY CONVNET
        tfcnn = Sequential()
        # feature maps = 40
        # 8x4 time-frequency filter (goes along both time and frequency axes)
        color_scale = 1
        input_size = (frame_width,num_features,color_scale)
        tfcnn.add(Conv2D(num_feature_columns, kernel_size=(8,4), activation='relu'))
        #non-overlapping pool_size 3x3
        tfcnn.add(MaxPooling2D(pool_size=(3,3)))
        tfcnn.add(Dropout(0.25))
        tfcnn.add(Flatten())
        
        #prepare LSTM
        tfcnn_lstm = Sequential()
        timestep = samples_per_utterance//frame_width
        tfcnn_lstm.add(TimeDistributed(tfcnn,input_shape=(timestep,frame_width,num_feature_columns,color_scale)))
        tfcnn_lstm.add(LSTM(timestep)) #num timesteps
        tfcnn_lstm.add(Dense(len(labels_present),activation="softmax")) # binary = "sigmoid"; multiple classification = "softmax"
        
        
        print(tfcnn_lstm.summary())
        
        
        #compile model
        tfcnn_lstm.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy']) # binary = "binary_crossentropy", multiple (one-hot-encoded) = "categorical_crossentropy"; multiple (integer encoded) = "sparse_categorical_crossentropy" 
        #train model
        epochs = 5
        tfcnn_lstm.fit(X_train, y_train, epochs=epochs, validation_split = 0.15)
        
        
        score = tfcnn_lstm.evaluate(X_test,y_test,verbose=1)
        acc = round(score[1]*100,2)
        
        print("Model Accuracy on test data:")
        print(acc)
        logging.info("Model Accuracy on TEST data: {}".format(acc))
        
        
        modelname = "CNN_LSTM_{}_{}_{}_{}features_{}recordings_{}epochs_{}acc".format(session_name,database,tablename,num_features,num_utterances,epochs,acc)
        print('Saving Model')
        tfcnn_lstm.save(modelname+'.h5')
        print('Done!')
        print("\n\nModel saved as:\n{}".format(modelname))
        

    except ExitApp:
        print("Have a good day!")
        logging.info("User exited app.")
    except Error as e:
        logging.exception("Database error: {}".format(e))
    except Exception as e:
        logging.exception("Error occurred: {}".format(e))
    finally:
        end = time.time()
        duration = round((end-start)/3600,3)
        msg = "Duration: {} hours".format(duration)
        logging.info(msg)
        print(msg)


if __name__=="__main__":
    main(script_purpose="speech_feature_prep_train_model_speech_recognition",database="speech_commands.db",tablename="fbank_delta_120_no_noise")
