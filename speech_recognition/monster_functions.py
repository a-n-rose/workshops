import numpy as np
from errors import TotalSamplesNotAlignedSpeakerSamples


def fill_matrix_samples_zero_padded(matrix2fill, row_id, data_supplied, indices, utterance_label, len_samps_per_id, frame_width):
    '''
    This function fills a matrix full of zeros with the same number of rows dedicated to 
    each speaker. 
    
    If the speaker has too many samples, not all will be included. 
    If the speaker has too few samples, only the samples that will complete a full window will
    be included; the rest will be replaced with zeros/zero padded.
    
    
    1) I need the len of matrix, to be fully divisible by len_samps_per_id 
    
    2) len_samps_per_id needs to be divisible by context_window_total (i.e. context_window_size * 2 + 1)
    
    2) label column assumed to be last column of matrix2fill
    
    #mini test scenario... need to put this into unittests
    empty_matrix = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    
    #each id has 3 rows
    data_supplied = np.array([[2,2,3],[2,2,3],[2,2,3],[2,2,3],[2,2,3],[2,2,3],[4,4,5],[4,4,5],[4,4,5],[4,4,5],[4,4,5],[4,4,5],[4,4,5],[4,4,5],[4,4,5],[4,4,5],[4,4,5],[4,4,5],[4,4,5],[4,4,5],[1,1,7],[1,1,7],[1,1,7],[1,1,7],[1,1,7],[1,1,7],[1,1,7],[1,1,7],[1,1,7],[1,1,7]])
    
    indices_too_few = [0,1,2,3,4,5] #too few samples (6/10)  (total_window_size = 5) 
    
    label_too_few = 1
    
    indices_too_many = [6,7,8,9,10,11,12,13,14,15,16,17,18,19] #too many (14/10) (total_window_size = 5) 
    
    label_too_many = 0
    
    indices_just_right = [20,21,22,23,24,25,26,27,28,29] #10/10 (total_window_size = 5) 
    
    label_just_right = 1
    
    len_samps_per_id = 10
    
    label_for_zeropadded_rows = 2
    
    empty_matrices should be:
    
    row_id = 0 --> row_id = 10
    
    matrix_too_few = np.array([[2,2,3,1],[2,2,3,1],[2,2,3,1],[2,2,3,1],[2,2,3,1],[0,0,0,2],[0,0,0,2],[0,0,0,2],[0,0,0,2],[0,0,0,2],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    
    row_id = 10 --> row_id = 20
    matrix_too_many = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[4,4,5,0],[4,4,5,0],[4,4,5,0],[4,4,5,0],[4,4,5,0],[4,4,5,0],[4,4,5,0],[4,4,5,0],[4,4,5,0],[4,4,5,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    
    row_id = 20 --> row_id = 30
    matrix_too_many = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[1,1,7,1],[1,1,7,1],[1,1,7,1],[1,1,7,1],[1,1,7,1],[1,1,7,1],[1,1,7,1],[1,1,7,1],[1,1,7,1],[1,1,7,1]])
    '''
    try:
        
        tot_samps_utterance = len(indices)
        tot_samps_sets = tot_samps_utterance//frame_width
        tot_samps_possible = tot_samps_sets * frame_width
        
        if tot_samps_possible > len_samps_per_id:
            tot_samps_possible = len_samps_per_id
            indices = indices[:tot_samps_possible]
        
        #keep track of the samples put into the new matrix
        #don't want samples to exceed amount set by variable 'len_samps_per_id'
        samp_count = 0
        
        for index in indices:
            
            #samples only get added to matrix if fewer than max number
            if samp_count < len_samps_per_id and row_id < len(matrix2fill):
                new_row = np.append(data_supplied[index],utterance_label)
                matrix2fill[row_id] = new_row
                samp_count += 1
                row_id += 1
            else:
                if row_id >= len(matrix2fill):
                    raise TotalSamplesNotAlignedSpeakerSamples("Row id exceeds length of matrix to fill.")
            # if all user samples used, but fewer samples put in matrix than max amount, zero pad
            if samp_count < len_samps_per_id and samp_count == tot_samps_possible:
                zero_padded = len_samps_per_id - samp_count
                
                if np.modf(zero_padded/frame_width)[0] != 0.0:
                    raise TotalSamplesNotAlignedSpeakerSamples("Zero padded rows don't match window frame size")
                
                for row in range(zero_padded):
                    #leave zeros, keep track of row_id and samp_count
                    row_id += 1
                    samp_count += 1
            
            #once all necessary samples put into matrix, leave loop and continue w next utterance 
            elif samp_count == len_samps_per_id:
                break
            
            #samp_count should not be greater than len_samps_per_id... if it is, something went wrong.
            elif samp_count > len_samps_per_id:
                raise TotalSamplesNotAlignedSpeakerSamples("More samples collected than max amount")

    except TotalSamplesNotAlignedSpeakerSamples as e:
        print(e)
        row_id = False
    
    return matrix2fill, row_id

