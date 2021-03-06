
'''
Script outline

1) load data 
expects title of table to contain: 
- 'mfcc' or fbank 
- the number of features
- optionaly: 'pitch' or 'delta' if the table has those features

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
        separator = "*"*80
        logging.info(separator)
        logging.info("RUNNING SCRIPT: \n\n{}".format(current_filename))
        logging.info("SESSION: \n\n{}".format(session_name))
        
        ######################################################################
        
        #load data
        logging.info("Loading data from \nDatabase: {}\nTable: {}".format(database,tablename))
        data = user_input.load_data(database,tablename)
        logging.info("Data successfully loaded")
        end_loaded_data = time.time()

        #!!!!necessary variables for user to set!!!!!
        #~these set most of the subsequent variables
        id_col_index = 2 #index 0 --> sample ID, index 1 --> speaker ID
        context_window_size = 9
        frame_width = context_window_size*2+1
        
        #if the data contains column w frequency info, assume it is the second to last column
        #also assumes features start after the relevant id column
        if 'pitch' in tablename:
            features_start_stop_index = [id_col_index+1,-2]
        else:
            features_start_stop_index = [id_col_index+1,-1]
        #assumes last column is the label column
        label_col_index = [-1]
        
        
        #add feature columns based on which features are to be expected
        num_features, num_feature_columns = feature_column_prep(tablename)
        
        print("The original number of features: {}".format(num_features))
        print("Total feature columns: {}".format(num_feature_columns))
    
        logging.info("Column index for each recording/speaker ID set at: {}".format(id_col_index))
        logging.info("Number of original features (e.g. MFCCs or FBANK energy features): {}".format(num_features))
        logging.info("Number of total features (e.g. derivatives, pitch): {}".format(num_feature_columns))
        logging.info("Set context window size: {}".format(context_window_size))
        logging.info("Frame width: {}".format(frame_width))
        
        
        ######################################################################

        start_data_prep = time.time()
        logging.info("Now prepping data for model training")
        #prep data
        #1) make sure each utterance has same number of samples;
        #if not, zeropad them so each has same number of samples
        data_zeropadded, samples_per_utterance, num_utterances, labels_present = featfun.prep_data(data,id_col_index,features_start_stop_index,label_col_index,num_feature_columns,frame_width,session_name)
        
        logging.info("Data has been zero-padded")
        logging.info("Shape of zero-padded data: {}".format(data_zeropadded.shape))
        logging.info("Fixed number of samples per utterance: {}".format(samples_per_utterance))
        logging.info("Number of utterances in data: {}".format(num_utterances))
        
        logging.info("Reshaping data to fit ConvNet + LSTM models")
        X, y = featfun.shape_data_dimensions_CNN_LSTM(data_zeropadded,samples_per_utterance,frame_width)
        logging.info("Done reshaping")
        logging.info("Shape of feature data (i.e. 'X'): {}".format(X.shape))
        logging.info("Shape of label data (i.e. 'y'): {}".format(y.shape))
        
        #separate X and y --> training and test datasets
        logging.info("Separating data into train and test datasets")
        test_size = 0.1
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = test_size)
        logging.info("Separated data. Test size = {}".format(test_size))
        logging.info("Shape of train data: \nX = {}\ny = {}".format(X_train.shape,y_train.shape))
        logging.info("Shape of test data: \nX = {}\ny = {}".format(X_test.shape,y_test.shape))
        ######################################################################
        
        #train the models!
        logging.info("Now initializing the model and beginning training")
        start_train = time.time()
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
        
        #set loss:
        #binary = "binary_crossentropy", multiple (one-hot-encoded) = "categorical_crossentropy"; multiple (integer encoded) = "sparse_categorical_crossentropy" 
        loss = "sparse_categorical_crossentropy"
        logging.info("Loss set at: '{}'".format(loss))
        
        #compile model
        tfcnn_lstm.compile(optimizer='adam',loss=loss,metrics=['accuracy'])
        #train model
        epochs = 300
        logging.info("Number of epochs set at: {}".format(epochs))
        
        model_train_name = "CNN_LSTM_training_{}".format(session_name)
        callback = [EarlyStopping(monitor='val_loss', patience=15, verbose=1), 
                    ReduceLROnPlateau(patience=5, verbose=1),
                    CSVLogger(filename='model_log/{}_log.csv'.format(model_train_name)),
                    ModelCheckpoint(filepath='bestmodel/bestmodel_{}.h5'.format(model_train_name), verbose=1, save_best_only=True)]
        
        history = tfcnn_lstm.fit(X_train, y_train, epochs=epochs, validation_split = 0.15, callbacks = callback)
        
        
        score = tfcnn_lstm.evaluate(X_test,y_test,verbose=1)
        acc = round(score[1]*100,2)
        
        print("Model Accuracy on test data:")
        print(acc)
        logging.info("Model Accuracy on TEST data: {}".format(acc))
        
        
        modelname = "CNN_LSTM_{}_{}_{}_{}recordings_{}epochs_{}acc".format(session_name,database,tablename,num_utterances,epochs,acc)
        print('Saving Model')
        tfcnn_lstm.save(modelname+'.h5')
        print('Done!')
        print("\n\nModel saved as:\n{}".format(modelname))
        
        print("Now saving history and plots")
        
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title("train vs validation loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train","validation"], loc="upper right")
        plt.savefig("{}_LOSS.png".format(modelname))
        
        plt.clf()
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title("train vs validation accuracy")
        plt.legend(["train","validation"], loc="upper right")
        plt.savefig("{}_ACCURACY.png".format(modelname))        

    except ExitApp:
        print("Have a good day!")
        logging.info("User exited app.")
    except Error as e:
        logging.exception("Database error: {}".format(e))
    except Exception as e:
        logging.exception("Error occurred: {}".format(e))
    finally:
        end = time.time()
        duration = round((end-start)/3600,2)
        msg1 = "Total Duration: {} hours".format(duration)
        logging.info(msg1)
        
        duration_load_data = round((end_loaded_data - start)/60,2)
        msg2 = "Duration to load data: {} min".format(duration_load_data)
        logging.info(msg2)
        
        duration_prep = round((start_train - start_data_prep)/60,2)
        msg3 = "Duration to prep data: {} min".format(duration_prep)
        logging.info(msg3)
        
        duration_train = round((end-start_train)/60,2)
        msg4 = "Duration to train models: {} min".format(duration_train)
        logging.info(msg4)


if __name__=="__main__":
    main(script_purpose="speech_feature_prep_train_model_speech_recognition",database="speech_features.db",tablename="fbank_pitch_20_no_noise_word")
