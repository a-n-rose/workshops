'''
explore feature extraction 
'''

import librosa
import get_speech_features as sf


def print_shape(matrix_name,matrix):
    if isinstance(matrix,tuple):
        matrix = matrix[0]
    elif isinstance(matrix,list):
        shape = len(matrix)
    elif matrix.shape == ():
        shape = 1
        print(matrix)
    else:
        shape = matrix.shape
        
    print("Shape of {}: {}".format(matrix_name,shape))

if __name__=="__main__":

    filename = "speech_modelready_aislyn.wav"
    sr = 16000
    y,sr = librosa.load(filename,sr=sr)

    mfccs = sf.get_mfcc(y,sr)
    

    centroids = sf.get_spectral_centroid(y,sr)
    
    freq_mean = sf.get_freq_mean(y,sr)
    freq_sd = sf.get_freq_sd(y,sr)
    freq_median = sf.get_freq_median(y,sr)
    freq_mode = sf.get_freq_mode(y,sr)
    
    domfreq_mean = sf.get_domfreq_mean(y,sr)
    domfreq_min = sf.get_domfreq_min(y,sr)
    domfreq_max = sf.get_domfreq_max(y,sr)
    domfreq_range = sf.get_domfreq_range(y,sr)
    domfreq_median = sf.get_domfreq_median(y,sr)
    domfreq_mode = sf.get_domfreq_mode(y,sr)
    domfreq = sf.get_domfreq(y,sr)
    domfreq_first_quartile, domfreq_third_quartile, domfreq_interquartile_range = sf.get_1st_3rd_inter_quartile_range(domfreq)
    
    fundfreq = sf.get_fundfreq(y,sr)
    fundfreq_mean = sf.get_fundfreq_mean(y,sr)
    fundfreq_min = sf.get_fundfreq_min(y,sr)
    fundfreq_max = sf.get_fundfreq_max(y,sr)
    fundfreq_first_quartile,fundfreq_third_quartile,fundfreq_interquartile_range = sf.get_1st_3rd_inter_quartile_range(fundfreq)
    
    fundfreq_notes = sf.get_notes_from_freq(fundfreq)
    print("\n\nFundamental Frequency Notes:")
    print(fundfreq_notes)
    domfreq_notes = sf.get_notes_from_freq(domfreq)
    print("\n\nDominant Frequency Notes:")
    print(domfreq_notes)
    print("\n\n")
    
    
    skew_mfcc = sf.get_spectral_skewness(mfccs)
    kurtosis_mfcc = sf.get_spectral_kurtosis(mfccs)
    entropy_mfcc = sf.get_spectral_entropy(mfccs)
    
    
    x = entropy_mfcc
    print(x)
    
    

    features_list = [("MFCCs",mfccs),("Centroids",centroids),("Frequency Mean",freq_mean),("Frequency Standard Deviation",freq_sd),("Frequency Median",freq_median),("Frequency Mode",freq_mode),("Dominant Frequency Mean",domfreq_mean),("Dominant Frequency Minimum",domfreq_min),("Dominant Frequency Maximum",domfreq_max),("Dominant Frequency Range",domfreq_range),("Dominant Frequency Median",domfreq_median),("Dominant Frequency Mode",domfreq_mode),("Dominant Frequency",domfreq),("Dominant Frequency 1st Quartile",domfreq_first_quartile),("Dominant Frequency 3rd Quartile",domfreq_third_quartile),("Dominant Frequency Interquartile Range",domfreq_interquartile_range),("Fundamental Frequency Mean",fundfreq_mean),("Fundamental Frequency Minimum",fundfreq_min),("Fundamental Frequency Maximum",fundfreq_max),("Fundamental Frequency",fundfreq),("Fundamental Frequency 1st Quartile",fundfreq_first_quartile),("Fundamental Frequency 3rd Quartile",fundfreq_third_quartile),("Fundamental Frequency Interquartile Range",fundfreq_interquartile_range),("MFCC Skew",skew_mfcc),("MFCC Kurtosis",kurtosis_mfcc),("MFCC Entropy",entropy_mfcc)] 
    

    
    for item in features_list:
        print_shape(item[0],item[1])
