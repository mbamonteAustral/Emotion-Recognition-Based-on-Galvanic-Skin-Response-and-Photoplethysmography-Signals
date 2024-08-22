
#%%----------------------------------------------------------------

import numpy as np
import random as python_random
import socket

# ---------------  import constants and functions:
# https://stackoverflow.com/questions/15514593/importerror-no-module-named-when-trying-to-run-python-script
import sys, os

# https://stackoverflow.com/questions/8663076/python-best-way-to-add-to-sys-path-relative-to-the-current-running-script
from definitions import *
from functions import * #proyect's functions / defs

def get_dataframes_one_participant(participant,window_seconds, overlap_percentage):

    # https://keras.io/getting_started/faq/#how-to-do-hyperparameter-tuning-with-keras
    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(123)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    python_random.seed(123)


    try:
        participant
        print("participant: ", participant)
        participant = str(participant)
    except NameError:
        # assign a participant in order to run the whole script:
        participant = str(11)  # original = 28 
            

        print("           ")
        print("-------  ----------------------------------")
        print("-----------------------------------------")
        print("WARNING:")
        print("No participant was passed as argument: this file will be used instead: ", r"C:\Users\mbamo\Desktop\Datasets\K-EmoCon\data for vpn pc\E4_" + r"BVP_p" + participant + r"t.xlsx")
        # print("No participant was passed as argument: this file will be used instead: ", fname + r"BVP_p" + participant + r"t.xlsx")
        print("-----------------------------------------")
        print("-----------------------------------------")
        print("           ")

    # ---  Load participant Data


    if platform == "linux" or platform == "linux2":

        # current_folder=$(pwd)
        # file_path="$current_folder/subfolder_name/file_name"

        # physiological_base = r"./physiological/sub_"
        # annotation_base = r"./annotations/sub_"
        # seqs_order_num_base = r"seqs_"
        physiological_base = r"/home/marcos/datasets/case/physiological/sub_"
        annotation_base = r"/home/marcos/datasets/case/annotations/sub_"
        seqs_order_num_base = r"//home/marcos/datasets/case/seqs_"
    elif platform == "win32":
        if socket.gethostname() == "LAPTOP-R7AHG17P":  # pc gamer (Javier)
            print("corriendo en PC gamer:")
            physiological_base = r"C:\Users\Javier\Desktop\CASE_full\CASE_full\data\interpolated\physiological\sub_"
            annotation_base = r"C:\Users\Javier\Desktop\CASE_full\CASE_full\data\\interpolated\annotations\sub_"
            seqs_order_num_base = r"C:\Users\Javier\Desktop\CASE_full\CASE_full\metadata\seqs_"
            # Set the default encoding
            sys.stdout.reconfigure(encoding='utf-8')
        else:
            physiological_base = r"C:\Users\mbamo\Desktop\Datasets\CASE\interpolated\physiological\sub_"
            annotation_base = r"C:\Users\mbamo\Desktop\Datasets\CASE\interpolated\annotations\sub_"
            seqs_order_num_base = r"C:\Users\mbamo\Desktop\Datasets\CASE\metadata\seqs_"

    # Specify the path to your text file
    physiological_file = physiological_base + participant + '.csv'
    annotation_file = annotation_base + participant + '.csv'
    seqs_order_num_file = seqs_order_num_base + 'order_num_csv' + '.csv'

    # Use numpy's loadtxt function to load the data
    data = pd.read_csv(physiological_file)
    seqs_order_num = pd.read_csv(seqs_order_num_file)

    # time_vector = data['daqtime']/(1000*60)  # [minutes] 
    time_vector = data['daqtime']  # [ms] 

    # load annotations
    annotations = pd.read_csv(annotation_file)
    valence = annotations['valence']
    arousal = annotations['arousal']

    stimulus_tag_list = [map_video_to_tag(i) for i in data['video']] 

    data['tag'] = stimulus_tag_list  # dataframe now contains the stimulus tag of each video

    [ gsr_mc, ppg_mc ] = baseline_mean_centered_normalization(data, show_plot=0)

    gsr_signals, gsr_info = filter_gsr(gsr_mc, show_plot=0)

    ppg_signals = filter_ppg(ppg_mc, show_plot=0)

    #%%----------------------------------------------------------------


    def get_dataframes(window_seconds, overlap_percentage):
        # test windowing
        window_size = int(window_seconds*1000)  # ms
        # perc_overlap = 0.0
        perc_overlap = overlap_percentage
        step = int(window_size * (1-perc_overlap))
        overlap = True

        print("Window size (sec): ", window_size/1000)
        print("step (sec): ", step/1000)
        print("overlap: ", overlap)
        if overlap:
            print("perc. of overlap: ", (window_size-step)*100/window_size)
            print("overlap duration (sec): ", window_size*perc_overlap/1000)

        # Ensures windows_size and step are compatible to Deep Learning arquitecture and pooling operations
        window_size = nearest_multiple(window_size, multiple=16)
        step = nearest_multiple(step, multiple=16)  #ensures perc. of overlap is exact
                
        combined_df = perform_windowing(data, ppg_signals, gsr_signals, valence, arousal, seqs_order_num, participant, window_size, step, overlap, show_plot = 0)

        combined_df_median = median_voting(combined_df)

        return combined_df, combined_df_median


    #%%----------------------------------------------------------------
    # window_seconds = 11
    # overlap_percentage = 0.0
    combined_df, combined_df_median = get_dataframes(window_seconds, overlap_percentage)

    return combined_df, combined_df_median 


if __name__ == "__main__":
    window_seconds = 11
    overlap_percentage = 0.0
    participant = 3
    combined_df, combined_df_median = get_dataframes_one_participant(participant,window_seconds, overlap_percentage)

    bins = np.arange(1, 9.1, 0.1)  # The end value is 9.1 to include 9
    histogram_bins, bins, patches = plt.hist(combined_df["val_chunks"], bins=bins, edgecolor='black', alpha=0.7)
    plt.show()

    print(combined_df['val_chunks'].iloc[0].shape)



#%%----------------------------------------------------------------


# # index = 0
# # num_points = len(combined_df["aro_chunks"].iloc[index])
# # time_ms = range(num_points) # A simple range from 0 to num_points - 1

# def combined_df_all_participants():

#     window_seconds = 16
#     overlap_percentage = 0.0
#     combined_df, combined_df_median = get_dataframes(window_seconds, overlap_percentage)

#     bins = np.arange(1, 9.1, 0.1)  # The end value is 9.1 to include 9

#     window_seconds = 29
#     overlap_percentage = 0.0
#     combined_df1, combined_df_median1 = get_dataframes(window_seconds, overlap_percentage)

#     plt.figure()
#     plt.hist(combined_df["val_chunks"], bins=bins, edgecolor='black', alpha=0.7)
#     plt.show()

#     plt.figure()
#     plt.hist(combined_df1["val_chunks"], bins=bins, edgecolor='black', alpha=0.7)
#     plt.show()

#     combined_series = pd.concat([combined_df, combined_df1], ignore_index=True)



#     return combined_series


# combined_series = combined_df_all_participants()

# bins = np.arange(1, 9.1, 0.1)  # The end value is 9.1 to include 9

# histogram_bins, bins, patches = plt.hist(combined_series["val_chunks"], bins=bins, edgecolor='black', alpha=0.7)
# plt.show()
# #%%----------------------------------------------------------------

# # #%%----------------------------------------------------------------
# # num_of_windows = combined_df.shape[0]
# # precision = 2
# # prevalence_w_aro = 0
# # prevalence_sum_aro = 0
# # prevalence_w_val = 0
# # prevalence_sum_val = 0

# # # for index in range(num_of_windows):
# # for index in range(20):
# #     # prevalence_w_aro = 100*sum(1 for sample in combined_df["aro_chunks"].iloc[index] if round(sample,precision) == round(combined_df_median["aro_chunks"].iloc[index].iloc[0],precision))/combined_df["aro_chunks"].iloc[0].shape[0]
# #     # prevalence_w_val = 100*sum(1 for sample in combined_df["val_chunks"].iloc[index] if round(sample,precision) == round(combined_df_median["val_chunks"].iloc[index].iloc[0],precision))/combined_df["val_chunks"].iloc[0].shape[0]
    
# #     # # print("prevalence_w: ", prevalence_w_aro)
# #     # prevalence_sum_aro += prevalence_w_aro

# #     # # print("prevalence_w: ", prevalence_w_val)
# #     # prevalence_sum_val += prevalence_w_val

# #     # # plt.plot(combined_df["aro_chunks"].iloc[index], label='Window 1')
# #     # # plt.plot(combined_df_median["aro_chunks"].iloc[index], label='Median Window 1')
# #     # # plt.ylabel('Label')
# #     # # plt.xlabel('Time (ms)')

# #     plt.figure()
# #     plt.plot(combined_df["val_chunks"].iloc[index], label='Window 1')
# #     plt.plot(combined_df_median["val_chunks"].iloc[index], label='Median Window 1')
# #     plt.ylabel('Label')
# #     plt.xlabel('Time (ms)')
    
# # # prevalence_aro = prevalence_sum_aro / num_of_windows
# # # prevalence_val = prevalence_sum_val / num_of_windows

# # # # prevalence = prevalence_w / num_of_windows

# # # print("total Arousal prevalence in a participant: ", prevalence_aro)
# # # print("total Valence prevalence in a participant: ", prevalence_val)
# # # plt.savefig('participant_labels'+'.jpg', dpi=5000)
# # plt.show(block=True)



# #%%----------------------------------------------------------------
# plt.figure()
# index = 18
# # plt.hist(combined_df["val_chunks"].iloc[index], bins=90, edgecolor='black', alpha=0.7)
# # plt.hist(combined_df["val_chunks"], bins=90, edgecolor='black', alpha=0.7)

# bins = np.arange(1, 9.1, 0.1)  # The end value is 9.1 to include 9

# histogram_bins, bins, patches = plt.hist(combined_df["val_chunks"], bins=bins, edgecolor='black', alpha=0.7)

# # plt.savefig('histogram_hr'+'.jpg', dpi=600)


# #%%----------------------------------------------------------------
# # combined_series = pd.concat([s1, s2], ignore_index=True)

# data = [1,1,1,1,1,1,1,1,2,3,3,3,3,3,3,3,3]
# median = np.median(data)
# median
# median_vector = median* np.ones(np.size(data))
# plt.plot(data)
# plt.plot(median_vector)
# plt.title('clase A: 47%, clase B: 6%, clase C: 47% - y gana B: no es bueno')

# # plt.figure()
# # plt.hist(data)



