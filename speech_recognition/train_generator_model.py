'''
installed Pillow-5.4.1

1) try out paper implementations
2) try w ConvLSTM2D
'''

import time
import os
from sqlite3 import Error
import matplotlib.pyplot as plt

import numpy as np
#for training
from sklearn.model_selection import train_test_split
#for the models
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, LSTM, MaxPooling2D, Dropout, TimeDistributed, Reshape, Input, MaxPooling1D, Conv1D, Lambda, Conv3D, MaxPooling3D, ConvLSTM2D
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,CSVLogger,ModelCheckpoint

import user_input 
from errors import ExitApp, FeatureExtractionError
import feature_extraction_functions as featfun 

import logging
from my_logger import start_logging, get_date
logger = logging.getLogger(__name__)


#set variables
modelname = "CNN_LSTM_speech_recognition"
script_purpose = "Apply_ConvNetLSTM_chromagrams_generator"
batch_size = 6 #number of 19-framed segments of 120 features (and 3rgb values)
split = True
timestep = 6
frame_width = 19
num_features = 120
color_scale = 3 # if 'rgb', color_scale = 3, if 'grayscale', color_scale = 1
num_labels = 30
epochs = 40
loss_type = 'sgd'
if split:
    frame_width_total = frame_width
    shuffle_data = False
    data_png = "data_split"
else:
    frame_width_total = frame_width * timestep
    shuffle_data = True
    data_png = "data"
if color_scale == 1:
    color_str = "grayscale"
elif color_scale == 3:
    color_str = "rgb"



current_filename = os.path.basename(__file__)
session_name = get_date() #make sure this session has a unique identifier - link to model name and logging information
start = time.time()
start_logging(script_purpose)
separator = "*"*80
logging.info(separator)
logging.info("RUNNING SCRIPT: \n\n{}".format(current_filename))
logging.info("SESSION: \n\n{}".format(session_name))


#get total number of images for train, val, and test datasets
#need this for .fit_generator
num_train_images = featfun.get_num_images('{}_train'.format(data_png))
num_val_images = featfun.get_num_images('{}_val'.format(data_png))
num_test_images = featfun.get_num_images('{}_test'.format(data_png))
print(num_train_images)
print(num_val_images)
print(num_test_images)


##TIME-FREQUENCY CONVNET
#difference between: 1) padding="valid" 2) padding="same"
tfcnn = Sequential()
# feature maps = 40
# 8x4 time-frequency filter (goes along both time and frequency axes)
tfcnn.add(Conv2D(num_features, kernel_size=(8,4), activation='relu'))
#non-overlapping pool_size 3x3
tfcnn.add(MaxPooling2D(pool_size=(3,3)))
tfcnn.add(Dropout(0.25))
tfcnn.add(Flatten())

#prepare LSTM
model = Sequential()
model.add(TimeDistributed(tfcnn,input_shape=(timestep,num_features,frame_width,color_scale)))
model.add(LSTM(timestep,return_sequences=True,unroll=True)) #num timesteps
model.add(Flatten())
model.add(Dense(num_labels,activation="softmax")) # binary = "sigmoid"; multiple classification = "softmax"



print(model.summary())


loss = "categorical_crossentropy"
print("Loss set at: '{}'".format(loss))

#compile model
model.compile(optimizer=loss_type,loss=loss,metrics=['accuracy'])



#set up data generator:
train_datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator()

callback = [EarlyStopping(monitor='val_loss', patience=15, verbose=1), 
            ReduceLROnPlateau(patience=5, verbose=1),
            CSVLogger(filename='model_log/{}_log.csv'.format(modelname)),
            ModelCheckpoint(filepath='bestmodel/bestmodel_{}.h5'.format(modelname), verbose=1, save_best_only=True)]


train_generator = train_datagen.flow_from_directory(
        '{}_train'.format(data_png),
        target_size=(num_features, frame_width_total),
        batch_size=batch_size,
        color_mode = color_str,       
        class_mode='categorical',
        shuffle = shuffle_data,
        )

val_generator = test_datagen.flow_from_directory(
        '{}_val'.format(data_png),
        target_size=(num_features, frame_width_total),
        batch_size=batch_size,
        color_mode = color_str,        
        class_mode='categorical',
        shuffle = shuffle_data,
        )

test_generator = test_datagen.flow_from_directory(
        '{}_test'.format(data_png),
        target_size=(num_features, frame_width_total),
        batch_size=batch_size, #default is 32
        color_mode = color_str,      
        class_mode='categorical',
        shuffle = shuffle_data,
        )

#need to reshape data!! Keras need another dimension in the input..
#X_train = np.reshape(X_train, X_train.shape + (1,))
#https://github.com/keras-team/keras/issues/3386

def trainGeneratorFunc():
    while True:
        x, y = train_generator.next()
        x = np.reshape(x,(1,x.shape[0],x.shape[1],x.shape[2],color_scale))
        y = y[0,] #save just one from the 6 from the batchsize - they all have the same label
        y = np.reshape(y,(1,y.shape[0]))
        yield x, y
        
def valGeneratorFunc():
    while True:
        x, y = val_generator.next()
        x = np.reshape(x, (1,x.shape[0],x.shape[1],x.shape[2],color_scale))
        y = y[0,] #save just one from the 6 from the batchsize - they all have the same label
        y = np.reshape(y,(1,y.shape[0]))
        yield x, y
        
        
trainGenerator = trainGeneratorFunc()
valGenerator = valGeneratorFunc()
testGenerator = valGeneratorFunc()



model_name = "CNN_LSTM_generator_testing_{}epochs_{}trainimages_{}valimages".format(epochs,num_train_images,num_val_images)

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5)

csv_logging = CSVLogger(filename='model_log/{}_log.csv'.format(model_name))

checkpoint_callback = ModelCheckpoint(model_name+'.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')


history = model.fit_generator(
        trainGenerator,
        steps_per_epoch = (num_train_images//batch_size)/4,
        epochs = epochs,
        callbacks=[early_stopping_callback, checkpoint_callback],
        validation_data = valGenerator, 
        validation_steps = (num_val_images//batch_size)/4
        )


score = model.evaluate_generator(testGenerator, num_test_images//batch_size)

loss = round(score[0],2)
acc = round(score[1]*100,3)

msg="Model Accuracy on test data: {}%\nModel Loss on test data: {}".format(acc,loss)
print(msg)
logging.info(msg)

modelname_final = "CNN_LSTM_{}_{}_{}_{}TrVaTe_images_{}epochs_{}Optimizer_{}acc".format(session_name,num_train_images,num_val_images,num_test_images,epochs,loss_type,acc)
print('Saving Model')
model.save(modelname_final+'.h5')
print('Done!')
logging.info("\n\nModel saved as:\n{}".format(modelname_final))
print("\n\nModel saved as:\n{}".format(modelname_final))


print("Now saving history and plots")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("train vs validation loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train","validation"], loc="upper right")
plt.savefig("./graphs/{}_LOSS.png".format(modelname_final))

plt.clf()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title("train vs validation accuracy")
plt.legend(["train","validation"], loc="upper right")
plt.savefig("./graphs/{}_ACCURACY.png".format(modelname_final))
