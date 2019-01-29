#to work w Windows, Mac, and Linux:
from pathlib import Path, PurePath
#saving labels
import csv

#audio 
import librosa
import librosa.display
import matplotlib.pyplot as plt

#math/data prep
import numpy as np
import pandas as pd
import random
import math
import statistics
from scipy import stats
from sklearn.preprocessing import LabelEncoder
#from keras.utils.np_utils import to_categorical

from errors import FeatureExtractionError, TotalSamplesNotAlignedSpeakerSamples
#from monster_functions import fill_matrix_samples_zero_padded
import prep_noise as prep_data_vad_noise


def collect_labels(data_path):
    p = Path(data_path)
    labels = list(p.glob('*/'))
    labels = [PurePath(labels[i]) for i in range(len(labels))]
    labels = [x.parts[1] for x in labels if '_' not in x.parts[1]]
    labels = check_4_github_files(labels)
    return labels

def get_num_images(data_directory,image_format=None):
    if image_format is None:
        image_format = 'png'
    p = Path(data_directory)
    pics = list(p.glob('**/*.{}'.format(image_format)))
    return len(pics)
    

def check_4_github_files(labels_list):
    if 'README.md' in labels_list:
        labels_list.remove('README.md')
    if 'LICENSE' in labels_list:
        labels_list.remove('LICENSE')
    return labels_list
    

def collect_audio_and_labels(data_path):
    '''
    expects wavefiles to be in subdirectory: 'data'
    labels are expected to be the names of each subdirectory in 'data'
    speaker ids are expected to be the first section of each wavefile
    '''
    p = Path(data_path)
    waves = list(p.glob('**/*.wav'))
    #remove directories with "_" at the beginning
    paths = [PurePath(waves[i]) for i in range(len(waves)) if waves[i].parts[1][0]!="_"]
    labels = [j.parts[1] for j in paths ]
    return paths, labels
    

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
    utterance_count = 0
    for key, value in dictionary.items():
        utterance_count += 1
        speaker_id = key.split("_")[0]
        label = key.split("_")[1]
        #to iterate through all of the samples of each utterance
        #each word contains many feature sets/ samples
        for row in value:
            features = list(row)
            features.insert(0,utterance_count)
            features.insert(0,speaker_id)
            features.append(label)
            data_prepped.append(tuple(features))
    
    return data_prepped
 
 
def get_change_acceleration_rate(spectro_data):
    #first derivative = delta (rate of change)
    delta = librosa.feature.delta(spectro_data)
    #second derivative = delta delta (acceleration changes)
    delta_delta = librosa.feature.delta(spectro_data,order=2)
    return delta, delta_delta

def remove_silences(y):
    yt = librosa.effects.trim(y)
    return yt[0]

def apply_noise(y,sr,wavefile):
    #at random apply varying amounts of environment noise
    rand_scale = random.choice([0.0,0.25,0.5,0.75])
    if rand_scale > 0.0:
        #apply *known* environemt noise to signal
        total_length = len(y)/sr
        y_noise,sr = librosa.load(wavefile,sr=16000)
        envnoise_normalized = prep_data_vad_noise.normalize(y_noise)
        envnoise_scaled = prep_data_vad_noise.scale_noise(envnoise_normalized,rand_scale)
        envnoise_matched = prep_data_vad_noise.match_length(envnoise_scaled,sr,total_length)
        if len(envnoise_matched) != len(y):
            diff = int(len(y) - len(envnoise_matched))
            if diff < 0:
                envnoise_matched = envnoise_matched[:diff]
            else:
                envnoise_matched = np.append(envnoise_matched,np.zeros(diff,))
        y += envnoise_matched
    return y

def save_chroma(wavefile,split,frame_width,time_step,feature_type,num_features,num_feature_columns,noise,path_to_save_png,train=False,noise_wavefile=None,vad = False):

    y, sr = get_samps(wavefile)
    #y = remove_silences(y)
    if vad:
        y = prep_data_vad_noise.get_speech_samples(y,sr)
    if noise_wavefile:
        y = apply_noise(y,sr,noise_wavefile)
    extracted = []
    if "mfcc" in feature_type.lower():
        extracted.append("mfcc")
        features = get_mfcc(y,sr,num_mfcc=num_features)
        features -= (np.mean(features, axis=0) + 1e-8)
        if "delta" in feature_type.lower():
            delta, delta_delta = get_change_acceleration_rate(features)
            features = np.concatenate((features,delta,delta_delta),axis=1)
    elif "fbank" in feature_type.lower():
        extracted.append("fbank")
        features = get_mel_spectrogram(y,sr,num_mels = num_features)
        features -= (np.mean(features, axis=0) + 1e-8)
        if "delta" in feature_type.lower():
            delta, delta_delta = get_change_acceleration_rate(features)
            features = np.concatenate((features,delta,delta_delta),axis=1)
    elif "stft" in feature_type.lower():
        extracted.append("stft")
        features = get_stft(y,sr)
        features -= (np.mean(features, axis=0) + 1e-8)
        if "delta" in feature_type.lower():
            delta, delta_delta = get_change_acceleration_rate(features)
            features = np.concatenate((features,delta,delta_delta),axis=1)
    if features.shape[1] != num_feature_columns: 
        raise FeatureExtractionError("The file '{}' results in the incorrect  number of columns (should be {} columns): shape {}".format(wavefile,num_features,features.shape))
    

    features = np.transpose(features)
    name = Path(wavefile).parts[-1][:-4]
    
    if split:
        
        count = 0
        #while count <= time_step:
            #for i in range(0,features.shape[1],frame_width):
        for i in range(0,time_step*frame_width,frame_width):
            if i > features.shape[1]-1:
                features_step = np.zeros((num_feature_columns,frame_width))
            else:
                features_step = features[:,i:i+frame_width]
            
            if features_step.shape[1] != frame_width:
                diff = frame_width - features_step.shape[1]
                features_step = np.concatenate((features_step,np.zeros((num_feature_columns,diff))),axis=1)
            
            plt.clf()
            librosa.display.specshow(features_step)
            
            plt.tight_layout(pad=0)
            plt.savefig("{}{}_{}.png".format(path_to_save_png,name,count),pad_inches=0)
            count+=1
                #if count > time_step:
                    #break
    else:

        plt.clf()
        max_len = frame_width*time_step
        if features.shape[1] < max_len:
            diff = max_len - features.shape[1]
            features = np.concatenate((features,np.zeros((num_feature_columns,diff))),axis=1)
        librosa.display.specshow(features[:,:max_len])
        plt.tight_layout(pad=0)
        plt.savefig("{}{}.png".format(path_to_save_png,name))
        
    return True

def get_feats(wavefile,feature_type,num_features,delta=False,noise_wavefile = None,vad = False):
    y, sr = get_samps(wavefile)
    if vad:
        y = prep_data_vad_noise.get_speech_samples(y,sr)
    if noise_wavefile:
        y = apply_noise(y,sr,noise_wavefile)
    if delta:
        num_feature_columns = num_features*3
    else:
        num_feature_columns = num_features
        
    extracted = []
    if "mfcc" in feature_type.lower():
        extracted.append("mfcc")
        features = get_mfcc(y,sr,num_mfcc=num_features)
        features -= (np.mean(features, axis=0) + 1e-8)
        if "delta" in feature_type.lower():
            delta, delta_delta = get_change_acceleration_rate(features)
            features = np.concatenate((features,delta,delta_delta),axis=1)
    elif "fbank" in feature_type.lower():
        extracted.append("fbank")
        features = get_mel_spectrogram(y,sr,num_mels = num_features)
        features -= (np.mean(features, axis=0) + 1e-8)
        if "delta" in feature_type.lower():
            delta, delta_delta = get_change_acceleration_rate(features)
            features = np.concatenate((features,delta,delta_delta),axis=1)
    elif "stft" in feature_type.lower():
        extracted.append("stft")
        features = get_stft(y,sr)
        features -= (np.mean(features, axis=0) + 1e-8)
        if "delta" in feature_type.lower():
            delta, delta_delta = get_change_acceleration_rate(features)
            features = np.concatenate((features,delta,delta_delta),axis=1)
    if features.shape[1] != num_feature_columns: 
        raise FeatureExtractionError("The file '{}' results in the incorrect  number of columns (should be {} columns): shape {}".format(wavefile,num_features,features.shape))
    return features
    

def get_features(wavefile,feature_type,num_features,num_feature_columns,noise):
    if noise:
        '''
        ToDo:
        add option for adding noise to the data
        '''
        pass
    if isinstance(num_features,str):
        num_features = int(num_features)
    y, sr = get_samps(wavefile)
    extracted = []
    if "mfcc" in feature_type.lower():
        extracted.append("mfcc")
        features = get_mfcc(y,sr,num_mfcc=num_features)
        if "delta" in feature_type.lower():
            delta, delta_delta = get_change_acceleration_rate(features)
            features = np.concatenate((features,delta,delta_delta),axis=1)
    elif "fbank" in feature_type.lower():
        extracted.append("fbank")
        features = get_mel_spectrogram(y,sr,num_mels = num_features)
        if "delta" in feature_type.lower():
            delta, delta_delta = get_change_acceleration_rate(features)
            features = np.concatenate((features,delta,delta_delta),axis=1)
    if "pitch" in feature_type.lower():
        extracted.append("pitch")
        freq = np.array(get_domfreq(y,sr))
        #make into dimension matching features to concatenate them
        freq = freq.reshape(len(features),1)
        features = np.concatenate((features,freq),axis=1)

    if features.shape[1] != num_feature_columns: 
        raise FeatureExtractionError("The file '{}' results in the incorrect  number of columns (should be {} columns): shape {}".format(wavefile,num_features,features.shape))
    
    return features, extracted


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


def get_mel_spectrogram(y,sr,num_mels = None,window_size=None, window_shift=None):
    '''
    set values: default for mel spectrogram calculation (FBANK)
    - windows of 25ms 
    - window shifts of 10ms
    '''
    if num_mels is None:
        num_mels = 40
    if window_size is None:
        n_fft = int(0.025*sr)
    else:
        n_fft = int(window_size*0.001*sr)
    if window_shift is None:
        hop_length = int(0.010*sr)
    else:
        hop_length = int(window_shift*0.001*sr)
        
    fbank = librosa.feature.melspectrogram(y,sr,n_fft=n_fft,hop_length=hop_length,n_mels=num_mels)
    fbank = np.transpose(fbank)
    
    return fbank

def get_stft(y,sr,window_size=None, window_shift=None):
    if window_size is None:
        n_fft = int(0.025*sr)
    else:
        n_fft = int(window_size*0.001*sr)
    if window_shift is None:
        hop_length = int(0.010*sr)
    else:
        hop_length = int(window_shift*0.001*sr)
    stft = np.abs(librosa.feature.stft(y,sr,n_fft=n_fft,hop_length=hop_length)) #comes in complex numbers.. have to take absolute value
    stft = np.transpose(stft)
    
    return stft

def get_domfreq(y,sr):
    frequencies, magnitudes = get_freq_mag(y,sr)
    #select only frequencies with largest magnitude, i.e. dominant frequency
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
    #collect frequencies present and their magnitudes
    frequencies,magnitudes = librosa.piptrack(y,sr,hop_length=hop_length,n_fft=n_fft)
    frequencies = np.transpose(frequencies)
    magnitudes = np.transpose(magnitudes)
    
    return frequencies, magnitudes


def get_data_matrix(data,features_start_stop,many=True):
    if len(features_start_stop) > 1:
        data = data.iloc[:,features_start_stop[0]:features_start_stop[1]].values
    else:
        if many:
            data = data.iloc[:,features_start_stop[0]::].values
        else:
            data = data.iloc[:,features_start_stop[0]].values

    return data


def get_unique(pandas_series,limit):
    '''
    expects a pandas series and an integer
    '''
    if len(pandas_series) > limit:
        print("Iterating through data to save memory..")
        unique = []
        for item in pandas_series:
            if item not in unique:
                unique.append(item)
    else:
        print("Getting labels via pandas.Series.unique()")
        unique = pandas_series.unique()
    
    return unique
    
    
def get_col_id_name(data,col_id):
    cols = data.columns
    col_id_name = cols[col_id]

    return col_id_name


def get_ids(data,col_id):
    col_id_name = get_col_id_name(data,col_id)
    ids = data[col_id_name]
    ids_unique = get_unique(ids,1000000)
    
    return ids_unique, ids


def get_num_samples_per_id(ids,data,col_id):
    col_id_name = get_col_id_name(data,col_id)
    samples_list = []
    for id_num in ids:
        samples_list.append(sum(data[col_id_name]==id_num))
    
    return samples_list

def prep_data(data,id_col_index,features_start_stop_index,label_col_index,num_features,frame_width,session):
    data_df = pd.DataFrame(data)
    
    #find max num samples --> don't lose data, zero pad those that are fewer
    utterance_ids_unique, utterance_ids_repeated = get_ids(data_df,id_col_index)
    num_utterances = len(utterance_ids_unique)
    num_samps_per_utterance = get_num_samples_per_id(utterance_ids_unique,data_df,id_col_index)
    max_samps = max(num_samps_per_utterance)
    
    #need the samples to be fully divisible by the frame size: no partial frames
    samples_per_utterance_zero_padded = (max_samps//frame_width)*frame_width
    numrows_zeropadded_data = samples_per_utterance_zero_padded * num_utterances
    data_zeropadded = np.zeros((numrows_zeropadded_data,num_features+1))#+1 for labels column
    
    features = get_data_matrix(data_df,features_start_stop_index,many=True)
    labels = get_data_matrix(data_df,label_col_index,many=False)
    #change categorical labels to integers
    #save the label-integer pairings to .csv (session --> unique filename)
    y, labels_present = prep_categorical_labels(labels,session)
    
    # initialize row_id as 0, and it will get updated in the function
    # this helps me know that the right data is getting inserted in the right row of the new matrix
    row_id = 0
    try:
        if np.modf(numrows_zeropadded_data/samples_per_utterance_zero_padded)[0] != 0.0:
            raise TotalSamplesNotAlignedSpeakerSamples("Length of matrix does not align with total samples for each speaker")
        
        for i, id_num in enumerate(utterance_ids_unique):
            #PROBLEM
            #
            equal_utterance = utterance_ids_repeated == id_num
            indices = np.where(equal_utterance)[0]
            
            #get label for utterance
            label = y[indices[0]]
            
            data_zeropadded, row_id = fill_matrix_samples_zero_padded(data_zeropadded,row_id,features,indices,label,samples_per_utterance_zero_padded,frame_width)

    except TotalSamplesNotAlignedSpeakerSamples as e:
        print(e)
        data_zeropadded = None
    
    return data_zeropadded, samples_per_utterance_zero_padded, num_utterances, labels_present


def shape_data_dimensions_CNN_LSTM(data_zeropadded,samples_per_utterance_zero_padded, frame_width):
    '''
    prep data shape for ConvNet+LSTM:
    *assumes last column is label column
    
    shape = (num_speakers, num_sets_per_speaker; num_frames_per_set; num_features_per_frame; grayscale)
    
    If ConvNet and LSTM put together --> (66,32,19,120,1) if 66 speakers
    - ConvNet needs grayscale 
    - LSTM needs num_sets_per_speaker 
    
    If separate:
    - Convent needs grayscale (19,120,1)
    - LSTM needs number features in a series, i.e. 19 (19,120)
    '''
    
    #separate features from labels:
    features = data_zeropadded[:,:-1]
    print("Shape of features: {}".format(features.shape))
    labels = data_zeropadded[:,-1]
    print("Label index 0: {}".format(labels[0]))
    
    #number of frames within each set of utterance samples...
    num_frame_sets = samples_per_utterance_zero_padded//frame_width
    
    #number of sets (of utterance samples) in the data
    num_sets_samples = len(features)//num_frame_sets
    
    num_utterances = len(data_zeropadded)//samples_per_utterance_zero_padded
    
    #make sure only number of samples are included to make up complete context window frames of e.g. 19 frames (if context window frame == 9, 9 before and 9 after a central frame, so 9 * 2 + 1)
    check = len(data_zeropadded)//num_frame_sets
    if math.modf(check)[0] != 0.0:
        print("Extra Samples not properly removed")
    else:
        print("No extra samples found")
    
    #reshaping data to suit ConvNet + LSTM model training. 
    #see notes at top of function definition
    X = features.reshape(len(data_zeropadded)//samples_per_utterance_zero_padded,samples_per_utterance_zero_padded//frame_width,frame_width,features.shape[1],1)
    #collect labels only **once** per utterance 
    y_indices = list(range(0,len(labels),samples_per_utterance_zero_padded))
    y = labels[y_indices]
    #y = to_categorical(y)
    
    return X, y


def prep_categorical_labels(labels,session):
    #expects a numpy array
    y = pd.Series(labels)
    classes = get_unique(y,1000000)
    # encode class values as integers
    print("Fitting the encoder on the data..")
    encoder = LabelEncoder()
    encoder.fit(y)
    print("Encoding the labels..")
    classes_encoded = list(encoder.transform(classes))
    print("Saving labels..")
    save_class_labels(classes,classes_encoded,session)
    print("Encoding the data")
    y = encoder.transform(labels)
    
    return y, classes


def save_class_labels(class_labels,class_labels_encoded,session):
    dict_labels = {}
    for i, item in enumerate(class_labels_encoded):
        dict_labels[item] = class_labels[i]
    filename = 'dict_labels_{}.csv'.format(session)
    with open(filename,'w') as f:
        w = csv.writer(f)
        w.writerows(dict_labels.items())
    
    return None

def get_class_distribution(class_labels,labels_list): 
    dict_class_distribution = {}
    for label in class_labels:
        count = 0
        for label_item in labels_list:
            if label == label_item:
                count+=1
            dict_class_distribution[label] = count
    return dict_class_distribution
        

def get_min_samples_per_class(class_labels, labels_list):
    dict_class_distribution = get_class_distribution(class_labels,labels_list)
    min_val = (1000000, None)
    for key, value in dict_class_distribution.items():
        if value < min_val[0]:
            min_val = (value, key)
    return min_val


def get_max_nums_train_val_test(max_num_per_class):
    max_train = int(max_num_per_class*.8)
    max_val = int(max_num_per_class*.1)
    max_test = int(max_num_per_class*.1)
    sum_max_nums = max_train + max_val + max_test
    if max_num_per_class > sum_max_nums:
        diff = max_num_per_class - sum_max_nums
        max_train += diff
    return max_train, max_val, max_test

def get_train_val_test_indices(list_length):
    indices_ran = list(range(list_length))
    random.shuffle(indices_ran)
    train_len = int(list_length*.8)
    val_len = int(list_length*.1)
    test_len = int(list_length*.1)
    sum_indices = train_len + val_len + test_len
    if sum_indices != list_length:
        diff = list_length - sum_indices
        train_len += diff
    train_indices = []
    val_indices = []
    test_indices = []
    for i, item in enumerate(indices_ran):
        if i < train_len:
            train_indices.append(item)
        elif i >= train_len and i < train_len+val_len:
            val_indices.append(item)
        elif i >= train_len + val_len and i < list_length:
            test_indices.append(item)
    return train_indices, val_indices, test_indices


def make_dict_class_index(class_labels,labels_list):
    dict_class_index_list = {}
    for label in class_labels:
        dict_class_index_list[label] = []
        for i, label_item in enumerate(labels_list):
            if label == label_item:
                dict_class_index_list[label].append(i)
    return dict_class_index_list


def assign_indices_train_val_test(class_labels,dict_class_index,max_nums_train_val_test):
    dict_class_dataset_index_list = {}
    for label in class_labels:
        tot_indices = dict_class_index[label]
        tot_indices_copy = tot_indices.copy()
        random.shuffle(tot_indices_copy)
        train_indices = tot_indices_copy[:max_nums_train_val_test[0]]
        val_indices = tot_indices_copy[max_nums_train_val_test[0]:max_nums_train_val_test[0]+max_nums_train_val_test[1]]
        test_indices = tot_indices_copy[max_nums_train_val_test[0]+max_nums_train_val_test[1]:max_nums_train_val_test[0]+max_nums_train_val_test[1]+max_nums_train_val_test[2]]
        dict_class_dataset_index_list[label] = [train_indices,val_indices,test_indices]
    return dict_class_dataset_index_list


def get_feats4pickle(class_labels,dict_labels_encoded,data_filename4saving,max_num_samples,dict_class_dataset_index_list,paths_list,labels_list,feature_type,num_filters,time_step,frame_width,limit=None,delta=False,noise_wavefile=None,vad=False,dataset_index=0):
    msg = "\nExtracting features from {} samples. \nFeatures will be saved in the file {}".format(max_num_samples,data_filename4saving)
    print(msg)

    
    for label in class_labels:
        paths_list_dataset = []
        labels_list_dataset = []
        train_val_test_index_list = dict_class_dataset_index_list[label]
        #print(train_val_test_index_list[dataset_index])
        for k in train_val_test_index_list[dataset_index]:
            paths_list_dataset.append(paths_list[k])
            labels_list_dataset.append(labels_list[k])

            
        print("Extracting features from class: {}".format(label))
    
    #train_feats = Parallel(n_jobs=num_cores)(
        #delayed(all_feats)('./data/audio_train_trim/' + x + '.npy')
        #for x in tqdm(train.fname.values))

    #train_feats_df = pd.DataFrame(np.vstack(train_feats))


        for j, wav in enumerate(paths_list_dataset):
            if limit and j > limit:
                break
            else:
                feats = coll_feats_manage_timestep(time_step,frame_width,wav,feature_type,num_filters,delta=False, noise_wavefile=noise_wavefile,vad = True)
                np.save(data_filename4saving+"_"+label+"_.npy",feats)
    
    return True
    
def coll_feats_manage_timestep(time_step,frame_width,wav,feature_type,num_filters,delta=False,noise_wavefile=None,vad = True):
    feats = get_feats(wav,feature_type,num_filters,delta=False,noise_wavefile=noise_wavefile,vad = True)
    max_len = frame_width*time_step
    print(feats.shape)
    if feats.shape[0] < max_len:
        diff = max_len - feats.shape[0]
        feats = np.concatenate((feats,np.zeros((diff,feats.shape[1]))),axis=0)
    else:
        feats = feats[:max_len,:]
    print(feats.shape)
    return feats
        
    
