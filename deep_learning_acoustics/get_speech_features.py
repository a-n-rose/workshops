'''
ToDo:
unitests

Speech feature extraction script
extracts the following features (and then some):

mfcc: mel frequency cepstral coefficients
meanfreq: mean frequency (in kHz)
sd: standard deviation of frequency
median: median frequency (in kHz)
Q25: first quantile (in kHz)
Q75: third quantile (in kHz)
IQR: interquantile range (in kHz)
skew: skewness
kurt: kurtosis 
sp.ent: spectral entropy
sfm: spectral flatness
mode: mode frequency
centroid: frequency centroid
meanfun: mean fundamental frequency measured across acoustic signal
minfun: minimum fundamental frequency measured across acoustic signal
maxfun: maximum fundamental frequency measured across acoustic signal
meandom: mean of dominant frequency measured across acoustic signal
mindom: minimum of dominant frequency measured across acoustic signal
maxdom: maximum of dominant frequency measured across acoustic signal
dfrange: range of dominant frequency measured across acoustic signal
modindx: modulation index

'''
import numpy as np
import librosa
import statistics
from scipy import stats


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


def get_freq_mean(y,sr):
    frequencies, __ = get_freq_mag(y,sr)
    mean = np.array([np.mean(i) for i in frequencies])
    return mean

def get_freq_sd(y,sr):
    frequencies, __ = get_freq_mag(y,sr)
    std = np.array([np.std(i) for i in frequencies])
    return std

def get_freq_median(y,sr):
    '''
    perhaps better applied to pitch or dominant frequencies?
    
    For now just returns 0s
    '''
    frequencies, __ = get_freq_mag(y,sr)
    median = np.array([statistics.median(i) for i in frequencies])
    return median

def get_freq_mode(y,sr):
    '''
    perhaps better applied to pitch or dominant frequencies?
    
    For now just returns 0s
    '''
    frequencies, __ = get_freq_mag(y,sr)
    mode = np.array([stats.mode(i)[0] for i in frequencies])
    return mode


def get_fundfreq(y,sr):
    '''
    Not 100% sure about this...
    getting the lowest frequency
    '''
    frequencies, __ = get_freq_mag(y,sr)
    fund_freq = []
    for i in frequencies:
        freq = np.unique(i)
        if freq[0] != 0:
            print(freq[0])
            fund_freq.append(freq)
        else:
            if len(freq) > 1:
                fund_freq.append(freq[1])
    return fund_freq


def get_fundfreq_mean(y,sr):
    fund_freq_mean = np.mean(get_fundfreq(y,sr))
    return fund_freq_mean

def get_fundfreq_min(y,sr):
    fund_freq_min = min(get_fundfreq(y,sr))
    return fund_freq_min

def get_fundfreq_max(y,sr):
    fund_freq_max = max(get_fundfreq(y,sr))
    return fund_freq_max

def get_domfreq(y,sr):

    frequencies, magnitudes = get_freq_mag(y,sr)
    
    dom_freq_index = [np.argmax(item) for item in magnitudes]
    
    dom_freq = [frequencies[i][item] for i,item in enumerate(dom_freq_index) if frequencies[i][item] > 0]
    return dom_freq

def get_domfreq_mean(y,sr):
    dom_freq_mean = np.mean(get_domfreq(y,sr))
    return dom_freq_mean

def get_domfreq_min(y,sr):
    dom_freq_min = min(get_domfreq(y,sr))
    return dom_freq_min

def get_domfreq_max(y,sr):
    dom_freq_max = max(get_domfreq(y,sr))
    return dom_freq_max

def get_domfreq_range(y,sr):
    dom_freq_range = get_domfreq_max(y,sr) - get_domfreq_min(y,sr)
    return dom_freq_range

def get_domfreq_median(y,sr):
    dom_freq_median = statistics.median(get_domfreq(y,sr))
    return dom_freq_median

def get_domfreq_mode(y,sr):
    dom_freq = get_domfreq(y,sr)
    if len(np.unique(dom_freq))==len(dom_freq):
        dom_freq = np.array(dom_freq).astype(int)
    dom_freq_mode = statistics.mode(dom_freq)
    return dom_freq_mode

def get_1st_3rd_inter_quartile_range(frequency_list):
    q_1 = np.percentile(frequency_list, 25, interpolation='lower')
    q_3 = np.percentile(frequency_list, 75, interpolation='higher')
    q_range = abs(q_1 - q_3)
    return q_1,q_3,q_range

def get_spectral_skewness(spectrum):
    skew = stats.skew(spectrum)
    return skew

def get_spectral_kurtosis(spectrum):
    kurtosis = stats.kurtosis(spectrum)
    return kurtosis

def get_spectral_entropy(spectrum):
    entropy = stats.entropy(spectrum)
    return entropy

def get_spectral_flatness(y,sr,window_size=None, window_shift=None):
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
    flatness = librosa.feature.spectral_flatness(y,sr,n_fft=n_fft,hop_length=hop_length)
    flatness = np.transpose(flatness)
    return flatness

def get_spectral_centroid(y,sr,window_size=None, window_shift=None):
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
    centroids = librosa.feature.spectral_centroid(y,sr,n_fft=n_fft,hop_length=hop_length)
    centroids = np.transpose(centroids)
    return centroids

def get_freq_modulation_index():
    pass

def get_zero_crossings(y,sr):
    '''
    So far only first item in returned list has True value; rest False...
    '''
    zero_crossings = librosa.zero_crossings(y,sr)
    return zero_crossings
    
    
##exprimetnal

def get_cq_transform(y,sr):
    '''
    constant-Q transform 
    default values:
    - window shifts of 10ms
    '''   
    cqt = librosa.cqt(y,sr)
    cqt = np.transpose(cqt)
    return cqt


def get_chroma_cqt(y,sr):
    chroma_cqt = librosa.feature.chroma_cqt(y,sr)
    chroma_cqt = np.transpose(chroma_cqt)
    return chroma_cqt

def get_chroma_stft(y,sr,window_size=None, window_shift=None):
    '''
    set values: default for STFT calculation
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
        
    chroma_stft = librosa.feature.chroma_stft(y,sr,hop_length=hop_length,n_fft=n_fft)
    chroma_stft = np.transpose(chroma_stft)
    return chroma_stft


def get_chroma_cens(y,sr):
    '''
    "Chroma Energy Normalized" calculation

    defaults set by Librosa
    (y=None, sr=22050, C=None, hop_length=512, fmin=None, tuning=None, n_chroma=12, n_octaves=7, bins_per_octave=None, cqt_mode='full', window=None, norm=2, win_len_smooth=41)
    '''
    chroma_cens = librosa.feature.chroma_cens(y,sr)
    chroma_cens = np.transpose(chroma_cens)
    return chroma_cens

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

def get_rmse(y,sr,window_size=None, window_shift=None):
    '''
    set values: default for root-mean-squeare energy
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
        
    rmse = librosa.feature.melspectrogram(y,sr,frame_length=n_fft,hop_length=hop_length)
    rmse = np.transpose(rmse)
    return rmse

def get_spectral_bandwidth(y,sr,window_size=None, window_shift=None):
    '''
    set values: default 
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
        
    bw = librosa.feature.spectral_bandwidth(y,sr,n_fft=n_fft,hop_length=hop_length)
    bw = np.transpose(bw)
    return bw

def get_spectral_contrast(y,sr,window_size=None, window_shift=None):
    '''
    set values: default 
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
        
    contrast = librosa.feature.spectral_contrast(y,sr,n_fft=n_fft,hop_length=hop_length)
    contrast = np.transpose(contrast)
    return contrast

def get_spectral_rolloff(y,sr,window_size=None, window_shift=None):
    '''
    set values: default 
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
        
    rolloff = librosa.feature.spectral_rolloff(y,sr,n_fft=n_fft,hop_length=hop_length)
    rolloff = np.transpose(rolloff)
    return rolloff

def get_poly_features(y,sr,window_size=None, window_shift=None):
    '''
    set values: default 
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
        
    poly_features = librosa.feature.poly_features(y,sr,n_fft=n_fft,hop_length=hop_length)
    poly_features = np.transpose(poly_features)
    return poly_features

def get_tonnetz(y,sr):
    tonnetz = librosa.feature.tonnetz(y,sr)
    tonnetz = np.transpose(tonnetz)
    return tonnetz

def get_zero_crossing_rate(y,window_size=None, window_shift=None):
    '''
    set values: default 
    - windows of 25ms 
    - window shifts of 10ms
    '''
    try:
        if window_size is None:
            n_fft = int(0.025*sr)
        else:
            n_fft = int(window_size*0.001*sr)
        if window_shift is None:
            hop_length = int(0.010*sr)
        else:
            hop_length = int(window_shift*0.001*sr)
            
        zcr = librosa.feature.zero_crossing_rate(y,frame_length=n_fft,hop_length=hop_length)
        zcr = np.transpose(zcr)
        return zcr

    except TypeError as e:
        print(e)
        print("Did you include 'sampling rate'? That is not necessary for the function: \n\nget_zero_crossing_rate() \n.")

    return None

def get_notes_from_freq(frequency,octave=True, cents=True):
    '''
    input single or list of frequencies
    
    output single or list of notes
    
    defaults:
    octave = true --> shows which octave
    cents = true --> shows fractions
    NOTE: if cents == True, Octave also has to be True
    #'''
    #to prevent error:
    if cents == True:
        if octave == False:
            print("If 'cents=True', 'octave' also has to be true.")
            octave = True
    notes = librosa.hz_to_note(frequency,octave=octave,cents=cents)
    return notes


