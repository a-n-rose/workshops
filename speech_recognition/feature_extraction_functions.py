#to work w Windows, Mac, and Linux:
from pathlib import Path, PurePath

#math
import numpy as np
import statistics
from scipy import stats

#audio 
import librosa

from errors import FeatureExtractionError


def collect_audio_and_labels():
    '''
    expects wavefiles to be in subdirectory: 'data'
    labels are expected to be the names of each subdirectory in 'data'
    speaker ids are expected to be the first section of each wavefile
    '''
    p = Path('.')
    waves = list(p.glob('**/*.wav'))
    #remove words repeated by same speaker from collection
    x = [PurePath(waves[i]) for i in range(len(waves)) if waves[i].parts[-1][-5]==str(0)]
    y = [j.parts for j in x]
    return x, y
    

def get_speaker_id(path_label):
    '''
    A speaker provides input for more than just one word
    this allows the speaker to be put into dictionary without
    causing 'KeyError'
    '''
    speaker_id_info = path_label[-1].split("_")
    speaker_id = speaker_id_info[0]
    speaker_id_label = speaker_id+ "_" + path_label[1]
    if speaker_id_info[-1][0] != str(0):
        print("Problem with {}".format(path_label))
        print("Perhaps the speaker {} has already said the word '{}'.".format(speaker_id,path_label[1]))
        logging.info("Repeated speaker: {}".format(path_label))
        return None
    return speaker_id_label

def organize_data(dictionary, path_labels, features):
    '''
    index corresponds where we are in labels and paths
    
    expects a path_labels as list of tuples:
    index 0 --> folder "data"
    index 1 --> label for the speaker
    index 3 --> wavefile name of which contains the speaker ID in first section
    '''
    speaker_id_label = get_speaker_id(path_labels)
    if speaker_id_label:
        if speaker_id_label not in dictionary:
            dictionary[speaker_id_label] = features

    return dictionary

def prep_data4sql(dictionary):
    '''
    input:
    expects key to have speaker id and label, separated w "_"
    expects values to be the features used to train model, in numpy array
    
    returns:
    list of tuples. the tuples correspond to sql columns:
    index 0 --> speaker id
    index ... --> features
    index -1 --> label
    '''
    data_prepped = []
    for key, value in dictionary.items():
        speaker_id = key.split("_")[0]
        label = key.split("_")[1]
        
        #to iterate through all of the samples of each utterance
        #each word contains many feature sets/ samples
        for row in value:
            features = list(row)
            features.insert(0,speaker_id)
            features.append(label)
            data_prepped.append(tuple(features))
            
    return data_prepped
 
 
def get_features(wavefile,feature_type,num_features,noise):
    if noise:
        '''
        ToDo:
        add option for adding noise to the data
        '''
        pass
    y, sr = get_samps(wavefile)
    if "mfcc" in feature_type.lower():
        features = get_mfcc(y,sr)
    elif "fbank" in feature_type.lower():
        features = get_mel_spectrogram(y,sr)
    if "pitch" in feature_type.lower():
        freq = np.array(get_domfreq(y,sr))
        #make into dimension matching features to concatenate them
        freq = freq.reshape(len(features),1)
        features = np.concatenate((features,freq),axis=1)
    if features.shape[1] != num_features:
        raise FeatureExtractionError("The file '{}' results in the incorrect  number of columns: shape {}".format(wavefile,features.shape))
    return features
    
    
def get_samps(wavefile,sr=None,high_quality=None):
    if sr is None:
        sr = 16000
    if high_quality:
        quality = "kaiser_high"
    else:
        quality = "kaiser_fast"
    y, sr = librosa.load(wavefile,sr=sr,res_type=quality) 
    return y, sr

def get_mfcc(y,sr,num_mfcc=None,window_size=None, window_shift=None):
    '''
    set values: default for MFCCs extraction:
    - 40 MFCCs
    - windows of 25ms 
    - window shifts of 10ms
    '''
    if num_mfcc is None:
        num_mfcc = 40
    if window_size is None:
        n_fft = int(0.025*sr)
    else:
        n_fft = int(window_size*0.001*sr)
    if window_shift is None:
        hop_length = int(0.010*sr)
    else:
        hop_length = int(window_shift*0.001*sr)
        
    mfccs = librosa.feature.mfcc(y,sr,n_mfcc=num_mfcc,hop_length=hop_length,n_fft=n_fft)
    mfccs = np.transpose(mfccs)
    return mfccs

def get_mel_spectrogram(y,sr,window_size=None, window_shift=None):
    '''
    set values: default for mel spectrogram calculation (FBANK)
    - windows of 25ms 
    - window shifts of 10ms
    '''
    if window_size is None:
        n_fft = int(0.025*sr)
    else:
        n_fft = int(window_size*0.001*sr)
    if window_shift is None:
        hop_length = int(0.010*sr)
    else:
        hop_length = int(window_shift*0.001*sr)
        
    fbank = librosa.feature.melspectrogram(y,sr,n_fft=n_fft,hop_length=hop_length)
    fbank = np.transpose(fbank)
    return fbank


def get_domfreq(y,sr):

    frequencies, magnitudes = get_freq_mag(y,sr)
    
    dom_freq_index = [np.argmax(item) for item in magnitudes]
    
    dom_freq = [frequencies[i][item] for i,item in enumerate(dom_freq_index)]
    return dom_freq

def get_freq_mag(y,sr,window_size=None, window_shift=None):
    '''
    default values:
    - windows of 25ms 
    - window shifts of 10ms
    '''
    if window_size is None:
        n_fft = int(0.025*sr)
    else:
        n_fft = int(window_size*0.001*sr)
    if window_shift is None:
        hop_length = int(0.010*sr)
    else:
        hop_length = int(window_shift*0.001*sr)
    frequencies,magnitudes = librosa.piptrack(y,sr,hop_length=hop_length,n_fft=n_fft)
    
    frequencies = np.transpose(frequencies)
    magnitudes = np.transpose(magnitudes)

    return frequencies, magnitudes
