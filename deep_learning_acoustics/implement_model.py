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


def record_sound(sec):
    sr = 16000
    print("\n\nPlease say: 'Hallo, wie geht es ihnen?'")
    background = sd.rec(int(sec*sr),samplerate=sr,channels=1)
    sd.wait()
    return background, sr

def get_mfccs(filename,sr):
    y, sr = librosa.load(filename,sr, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y,sr,n_mfcc=40,hop_length=int(0.010*sr),n_fft=int(0.025*sr))
    mfccs = np.transpose(mfccs)
    return mfccs

if __name__=="__main__":
    num_features = 40
    frame_width = 19
    color_scale = 1
    
    print("\nType your firstname:\n")
    name = input()
    print("\nAre you ready to test the classifier? (Y/N):\n")
    ready = input()
    
    if isinstance(ready, str) and "y" in ready.lower():
        speaker_sample, sr = record_sound(6)
        
        filename = "./data/speaker_{}.wav".format(name.lower())
        sf.write(filename,speaker_sample,sr)
        print("Background noise successfully saved at the following location:\n\n{}".format(filename))
    #sr = 16000
    #filename = "./data/speaker_aislyn.wav"
        mfcc = get_mfccs(filename,sr)
            
        #load model:
        model_name = "female_male_speech_classifier_CNNLSTM_backgroundnoise_none3_98acc_samps_399"
        num_samples_per_speaker = int(model_name[-3:])
        print(num_samples_per_speaker)

        model = load_model(model_name+".h5")
            
        #now need to shape data for ConvNet+LSTM network:
        
        #prep the mfcc data for the model:
        num_samps_speaker = len(mfcc)
        print("\nNumber of samples from this speaker: {}".format(num_samps_speaker))
        
        print(mfcc.shape)
        mfcc = mfcc[:num_samples_per_speaker]
        print(mfcc.shape)
        
        num_series = num_samples_per_speaker//frame_width
        
        mfcc_classify = mfcc.reshape((1,num_series,frame_width,num_features,color_scale))
        print(mfcc_classify.shape)
        
        prediction = model.predict(mfcc_classify)
        print("\n\npredicted class:\n{}".format(prediction))
        if prediction > 0.5:
            class_assignment = "male"
            print("Speech is classified as {}".format(class_assignment.upper()))
        else:
            class_assignment = "female"
            print("Speech is classified as {}".format(class_assignment.upper()))
        print("\n\nIs this correct? (Y/N)\n")
        correct = input()
        if "y" in correct.lower():
            print("Great!!")
        elif "n" in correct.lower():
            print("Oops. My bad. \nI really don't think you sound {}. I just don't know how to handle all the different kinds of noise!\n\nHave a great day!".format(class_assignment))
