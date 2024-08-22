
#%% --------------------------------------------------
# Computes main dominance over all the participants

from dominancef import dominance
import numpy as np
from numpy import savetxt
from numpy import loadtxt
from sys import platform
from openpyxl import load_workbook
import time
import tensorflow as tf

overlap_percentage = [0.0, 0.25, 0.5]
window_size= [  1,    3,    5,    7,    9,    11,   13,  15,  16, 17,   19,   21,   23,   25,   27,  29]
participants = np.arange(1, 31)  # participant from 1 to 30


with tf.device('/CPU:0'):

    # Record the start time
    start_time = time.time()

    prev_arousal = np.zeros(np.shape(participants)[0], dtype=float) # prevalence for one particular participant (arousal)
    prev_valence = np.zeros(np.shape(participants)[0], dtype=float) # prevalence for one particular participant (valence)
    prevalence_wsize_aro = np.zeros(np.shape(window_size), dtype=float) # prevalence for all  participants for a particular window size (arousal)
    prevalence_wsize_val = np.zeros(np.shape(window_size), dtype=float) # prevalence for all participants for a particular window size (valence)
    prevalence_over_aro = np.zeros((np.shape(overlap_percentage)[0],np.shape(prevalence_wsize_aro)[0])) # each row has the prevalence for a particular overlap. Each columns: prevalence for a particular wsize) 
    prevalence_over_val = np.zeros((np.shape(overlap_percentage)[0],np.shape(prevalence_wsize_val)[0])) # each row has the prevalence for a particular overlap. Each columns: prevalence for a particular wsize) 
    
    # Standard deviation
    prevalence_wsize_aro_std = np.zeros(np.shape(window_size), dtype=float) # prevalence for all  participants for a particular window size (arousal)
    prevalence_wsize_val_std = np.zeros(np.shape(window_size), dtype=float) # prevalence for all participants for a particular window size (valence)
    prevalence_over_aro_std = np.zeros((np.shape(overlap_percentage)[0],np.shape(prevalence_wsize_aro)[0])) # each row has the prevalence for a particular overlap. Each columns: prevalence for a particular wsize) 
    prevalence_over_val_std = np.zeros((np.shape(overlap_percentage)[0],np.shape(prevalence_wsize_val)[0])) # each row has the prevalence for a particular overlap. Each columns: prevalence for a particular wsize) 
    

    for ov_perc_index in range(np.shape(overlap_percentage)[0]):
        for wsize_index in range(np.shape(window_size)[0]):
            for part_index in range(np.shape(participants)[0]):
                prevalence_aro, prevalence_val = dominance(participants[part_index], window_size[wsize_index], overlap_percentage[ov_perc_index])
                prev_arousal[part_index] = prevalence_aro
                prev_valence[part_index] = prevalence_val
                # print("------------------------------------")
                # print("overlap_percentage: ", overlap_percentage[ov_perc_index])
                # print("window size: ", window_size[wsize_index])
                # print("participant: ", participants[part_index])
                # print("------------------------------------")
            
            # print("------------------------------------")
            # print("For Window size: ", window_size[wsize_index])
            # print("prev_arousal: ", prev_arousal)
            # print("prev_valence: ", prev_valence)
            # print("------------------------------------")
            prevalence_wsize_aro[wsize_index] = np.nanmean(prev_arousal) 
            prevalence_wsize_val[wsize_index] = np.nanmean(prev_valence)

            prevalence_wsize_aro_std[wsize_index] = np.nanstd(prev_arousal) 
            prevalence_wsize_val_std[wsize_index] = np.nanstd(prev_valence)

        # print("------------------------------------")
        # print("For On perc of Overlap: ", overlap_percentage[ov_perc_index])
        # print("prevalence_wsize_aro: ", prevalence_wsize_aro)
        # print("prevalence_wsize_val: ", prevalence_wsize_val)
        # print("------------------------------------")
        prevalence_over_aro[ov_perc_index,:] = prevalence_wsize_aro #in each row is the prevalence for a particular overlap. columns: diff wsizes
        prevalence_over_val[ov_perc_index,:] = prevalence_wsize_val

        prevalence_over_aro_std[ov_perc_index,:] = prevalence_wsize_aro_std #in each row is the prevalence for a particular overlap. columns: diff wsizes
        prevalence_over_val_std[ov_perc_index,:] = prevalence_wsize_val_std
    
    print("------------------------------------")
    print("Final results: -----------")
    print("prevalence_over_aro: ", prevalence_over_aro)
    print("prevalence_over_val: ", prevalence_over_val)
    print("------------------------------------")


    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time  # in seconds

    print("Elapsed time:", elapsed_time/(60), "minutes")
    print("Elapsed time:", elapsed_time/(60*60), "hours")

    # %% -----------------------------------
    # save performance in .csv file:
    print("Saving dominance performance metrics in csv files...")
    savetxt('prevalence_over_aro.csv',prevalence_over_aro, delimiter=',')
    savetxt('prevalence_over_val.csv',prevalence_over_val, delimiter=',')
    savetxt('prevalence_over_aro_std.csv',prevalence_over_aro_std, delimiter=',')
    savetxt('prevalence_over_val_std.csv',prevalence_over_val_std, delimiter=',')
    
    # Introduce a delay of 5 seconds to give savetxt to really save each file and use it bellow.  
    time.sleep(5)

    print("Dominance Performance metrics already saved in csv files")

    