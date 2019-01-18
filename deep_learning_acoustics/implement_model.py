'''
Script to collect new speech and deploy trained model to classify new speech

'''
import numpy as np
import pandas as pd
import sounddevice as sd
import soundfile as sf
import librosa
import time
import random
import math

import keras
from keras.models import model_from_json
from keras.models import load_model

import speech_collection 
import get_speech_features as sf
from errors import ExitApp




if __name__=="__main__":
    try:
        sex = "female"
        
        num_features = 41
        num_mfcc = 40
        frame_width = 11
        color_scale = 1
        sampling_rate = 16000
        noise_filename = "background_noise.wav"
        if sex == "female":
            speech_filename = "speaker_aislyn_satz.wav"
            speech_nr_filename = "speech_modelready_aislyn11.wav"
        else:
            speech_filename = "speaker_thomas_satz.wav"
            speech_nr_filename = "speech_modelready_thomas9.wav"
            

        print("Testing classifier with {} speech:".format(sex))
        print("Press ENTER to start")
        ready = input()
        
        if ready != "":
            raise ExitApp()
        
        
        #background_noise = speech_collection.record(5,sampling_rate)
        #speech_collection.save_recording("noise_trash.wav",background_noise,sampling_rate)
        #print("Please press Enter and then say loudly and clearly: \n\n'Hallo, wie geht es Ihnen?'\n")
        #start = input()
        #if start != "":
        #    raise ExitApp()
        
        #speech = speech_collection.record(5,sampling_rate)
        #print("\nNow saving..")
        #speech_collection.save_recording("speech_trash.wav",speech,sampling_rate)
        
        print("\nNow extracting features..\n")
        y_speech, sr = librosa.load(speech_filename,sr = sampling_rate)
        #y_noise, sr = librosa.load("background_noise.wav",sr=sampling_rate)
        #speech_rednoise = speech_collection.vad_rednoise(y_speech,y_noise,sampling_rate)
        
        #speech_collection.save_recording(speech_nr_filename,speech_rednoise,sampling_rate)
        
        mfcc = sf.get_mfcc(y_speech,sampling_rate,num_mfcc)
        pitch = np.array(sf.get_domfreq(y_speech,sampling_rate))
        pitch = pitch.reshape(len(pitch),1)
        
        features = np.concatenate((mfcc,pitch),axis=1)
        print("\nClassifying speech..")

            
        #load model:
        model_name = "female_male_mfcc_domfreq_classifier_CNNLSTM_96acc_samps_407"
        num_samples_per_speaker = int(model_name[-3:])
        print(num_samples_per_speaker)

        model = load_model(model_name+".h5")
            
        #now need to shape data for ConvNet+LSTM network:
        print(features.shape)
        features = features[:num_samples_per_speaker]
        print(features.shape)
        
        num_series = num_samples_per_speaker//frame_width
        features_classify = features.reshape((1,num_series,frame_width,num_features,color_scale))
        print(features_classify.shape)
        
        prediction = model.predict(features_classify)
        print("\n\npredicted class:\n{}".format(prediction))
        if prediction > 0.60:
            print("\nThis is pretty clear.\n")
            class_assignment = "male"
        elif prediction <= 0.6 and prediction >= 0.4:
            print("\nThis is a close call... could go either way....\n")
            if prediction > 0.5:
                class_assignment = "male"
            elif prediction < 0.5:
                class_assignment = "female"
            else:
                class_assignment = "could not classify: either male or female"
        else:
            print("\nThis is pretty clear.\n")
            class_assignment = "female"
        
        print("Speech is classified as {}".format(class_assignment.upper()))
        print("\n\nIs this correct? (Y/N)\n")
        correct = input()
        if "y" in correct.lower():
            print("Great!!")
        elif "n" in correct.lower():
            print("Oops. My bad. \nI really don't think you sound {}. I just don't know how to handle all the different kinds of noise!\n\nHave a great day!".format(class_assignment))
    except ExitApp:
        print("\nHave a great day!\n")
