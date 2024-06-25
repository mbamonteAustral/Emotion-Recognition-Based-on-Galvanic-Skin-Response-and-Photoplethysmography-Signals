#%%----------------------------------------------------------------
# Plot annotation decision methods (plot median voting figures)

# import pickle
# from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from sys import platform
import pandas as pd
import neurokit2 as nk # Load the NeuroKit package
from definitions import *


# base as a function of the platform:
if platform == "linux" or platform == "linux2":
    physiological_base = r"E4_"
elif platform == "win32":
    physiological_base = r"C:\Users\mbamo\Desktop\Datasets\CASE\interpolated\physiological\sub_"
    annotation_base = r"C:\Users\mbamo\Desktop\Datasets\CASE\interpolated\annotations\sub_"
    seqs_order_num_base = r"C:\Users\mbamo\Desktop\Datasets\CASE\metadata\seqs_"

# Specify the path to your text file
participant = '10'  #29
physiological_file = physiological_base + participant + '.csv'
annotation_file = annotation_base + participant + '.csv'
seqs_order_num_file = seqs_order_num_base + 'order_num_csv' '.csv'

# Use numpy's loadtxt function to load the data
data = pd.read_csv(physiological_file)

# time_vector = data['daqtime']/(1000*60)  # [minutes] 
time_vector = data['daqtime']  # [ms] 

# load annotations
annotations = pd.read_csv(annotation_file)
valence = annotations['valence']
arousal = annotations['arousal' ]

seqs_order_num = pd.read_csv(seqs_order_num_file)




#%% ----------------------------------------------------------------
# Majority agreement protocol

# Function to find the majority element (must be an element repeated contiguously in time)
def findMajority(arr, n):
    candidate = np.nan  # or -1
    votes = 0

    # Finding the majority candidate
    for i in range(n):
        if votes == 0:
            candidate = arr[i]
            votes = 1
        else:
            if arr[i] == candidate:
                votes += 1
            else:
                votes -= 1
    count = 0

    # Checking if the majority candidate occurs more than n/2 times
    for i in range(n):
        if arr[i] == candidate:
            count += 1
    if count > n // 2:
        return candidate
    else:
        # return -1
        return np.nan


#%%----------------------------------------------------------------

def compare_annotation_chunck_decision_methods(annotation_segments, method = "all", chunck_number = 1):
    """
    Compare the different annotation chunck decision methods.
    Method = "median", "mean", "all". "all": compare mean and median.
    "Majority": It employes the Boyer–Moore majority vote algorithm. This methods is computed always, independent of the chosen method.
    """

    if method == "median":

        median = np.median(annotation_segments[chunck_number][:])
        median_array = np.ones((len(annotation_segments[chunck_number][:]),1))*median
        print("median: ", median)
        median_array_pd = pd.Series(np.ravel(median_array),index=annotation_segments[chunck_number][:].index)  # Get indexing of annotation_segments[chunk_number][:] pandas series to plot both time_series in the same plot and compare

        plt.plot(annotation_segments[chunck_number][:], label="label segment")
        plt.plot(median_array_pd, label="median")
        plt.legend()
        plt.show(block=True)

    elif method == "mean":

        mean = np.mean(annotation_segments[chunck_number][:])
        mean_array = np.ones((len(annotation_segments[chunck_number][:]),1))*mean
        print("mean: ", mean)
        mean_array_pd = pd.Series(np.ravel(mean_array),index=annotation_segments[chunck_number][:].index)  # Get indexing of annotation_segments[chunk_number][:] pandas series to plot both time_series in the same plot and compare

        plt.plot(annotation_segments[chunck_number][:], label="label segment")
        plt.plot(mean_array_pd, label="mean")
        plt.legend()
        plt.show(block=True)
    elif method == "all":

        median = np.median(annotation_segments[chunck_number][:])
        median_array = np.ones((len(annotation_segments[chunck_number][:]),1))*median
        
        median_array_pd = pd.Series(np.ravel(median_array),index=annotation_segments[chunck_number][:].index)  # Get indexing of annotation_segments[chunk_number][:] pandas series to plot both time_series in the same plot and compare
    
        mean = np.mean(annotation_segments[chunck_number][:])
        mean_array = np.ones((len(annotation_segments[chunck_number][:]),1))*mean
        
        mean_array_pd = pd.Series(np.ravel(mean_array),index=annotation_segments[chunck_number][:].index)  # Get indexing of annotation_segments[chunk_number][:] pandas series to plot both time_series in the same plot and compare

        n = len((annotation_segments[chunck_number][:]))
        majority = findMajority(list(annotation_segments[chunck_number][:]), n)
        majority_array = np.ones((np.shape(annotation_segments[chunck_number][:])[0],1))*majority

        # plot segment with a relative time (for plotting purposes)
        time_values = range(0,np.shape(annotation_segments[chunck_number][:])[0])
        time_values = np.array(time_values)/1000
        plt.plot(time_values,annotation_segments[chunck_number][:], label="annotation")
        plt.plot(time_values,mean_array_pd, label="mean")
        plt.plot(time_values,median_array_pd, label="median")
        plt.plot(time_values,majority_array, label="Boyer-Moore")
        
        plt.legend()
        plt.ylabel("Annotation")
        plt.xlabel("Time (seconds)")
        plt.savefig('median_voting'+str(chunck_number)+'.jpg', dpi=600)
        plt.show(block=True)    
        print("-------------------------------")
        print("median: ", median)
        print("mean: ", mean)


    print("-------------------------------")
    print("test voting ALG with measured annotations:")

    n = len((annotation_segments[chunck_number][:]))
    majority = findMajority(list(annotation_segments[chunck_number][:]), n)
    print(" The majority element is :" ,majority)



# a clear majority case:
compare_annotation_chunck_decision_methods(val_segments, method = "all", chunck_number = 40)

# a not clear majority case:
compare_annotation_chunck_decision_methods(val_segments, method = "all", chunck_number = 22)


# a not clear majority case:
# extreme case
# compare_annotation_chunck_decision_methods(val_segments, method = "all", chunck_number = 200)
compare_annotation_chunck_decision_methods(val_segments, method = "all", chunck_number = 27)



