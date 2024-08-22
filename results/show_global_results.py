"""
Show global simulated results
"""
#%% --------------------------------------------------
import numpy as np

# Sources (Authors' access only)

# 1) ML experiments planning - pc (Mind map)
# https://drive.mindmup.com/map/1hmYMXRqlrz2mcqbli2P8Ys0-OxRV-5rz

# 2) ML experiments planning - pc.xlsx (summary of results - Authors' access only)


#%%----------------------------------------------------------------

# Study the effects of different window duration sizes and percentages of overlap.

# bipartite scheme
# classic (threshold = 5)

# Data sources
# ML experiments planning - pc.xlsx (summary of results - Authors' access only)
# c_sldl_1_2_1
# c_sldl_1_2_2
# c_sldl_1_2_3
# c_sldl_1_2_4
# c_sldl_1_2_5
# c_sldl_1_2_6
# c_sldl_1_2_7
# c_sldl_1_2_8
# c_sldl_1_2_9
# c_sldl_1_2_10
# c_sldl_1_2_11
# c_sldl_1_2_12
# c_sldl_1_2_13
# c_sldl_1_2_14
# c_sldl_1_2_15
# c_sldl_1_2_16
# c_sldl_1_2_17
# c_sldl_1_2_18
# c_sldl_1_2_19
# c_sldl_1_2_20
# c_sldl_1_2_21
# c_sldl_1_2_22
# c_sldl_1_2_23
# c_sldl_1_2_24
# c_sldl_1_2_25
# c_sldl_1_2_26
# c_sldl_1_2_27
# c_sldl_1_2_28
# c_sldl_1_2_35
# c_sldl_1_2_36
# c_sldl_1_2_37
# c_sldl_1_2_38
# c_sldl_1_2_39
# c_sldl_1_2_40
# c_sldl_1_2_41
# c_sldl_1_2_42
# c_sldl_1_2_43
# c_sldl_1_2_44
# c_sldl_1_2_45
# c_sldl_1_2_46
# c_sldl_1_2_47
# c_sldl_1_2_48
# c_sldl_1_2_49
# c_sldl_1_2_50
# c_sldl_1_2_51
# c_sldl_1_2_52
# c_sldl_1_2_53
# c_sldl_1_2_54
# c_sldl_1_2_55
# c_sldl_1_2_56
# c_sldl_1_2_57
# c_sldl_1_2_58
# c_sldl_1_2_59
# c_sldl_1_2_60
# c_sldl_1_2_61


import matplotlib.pyplot as plt
import numpy as np

# ------------------------------ Arousal -------------------------------------

overlap0 =   [0.70, 0.71, 0.71, 0.71, 0.69, 0.70, 0.69, 0.67, 0.68, 0.69, 0.67, 0.66, 0.66, 0.65, 0.64]  
overlap25 =  [0.71, 0.73, 0.73, 0.72, 0.72, 0.73, 0.72, 0.70, 0.71, 0.71, 0.69, 0.7,  0.69, 0.68, 0.67] 
overlap50 =  [0.72, 0.75, 0.75, 0.75, 0.76, 0.761, 0.74, 0.74, 0.74, 0.74, 0.74, 0.74, 0.73, 0.74, 0.73]
window_size= [  1,    3,    5,    7,    9,    11,   13,  15,   17,   19,   21,   23,   25,   27,  29]


plt.plot(window_size, overlap0, 'o:', label='Overlap 0%')
plt.plot(window_size, overlap25, 'o:',label='Overlap 25%')
plt.plot(window_size, overlap50, 'o:', label='Overlap 50%')

# Add a vertical line at wsize = 3
plt.axvline(x=3, color='k', linestyle='--', label='$W_{TH}}$ Maximum window size')
plt.fill_between(window_size[:2], [0.8,0.8], [0.0,0.0], alpha=0.2)


plt.xlabel('Window size (sec)', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)

plt.xticks(window_size)  
plt.legend(loc='best', fontsize=10)
plt.ylim(0.63, 0.8)  # Adjust the limits as needed
plt.grid()
plt.savefig('window_size_effect_arousal'+'.jpg', dpi=600)
plt.show()



# --------------------------------  Valence -----------------------------------------

# Data
overlap0 =   [0.70, 0.71, 0.71, 0.71, 0.70, 0.70, 0.70, 0.70, 0.68,  0.69, 0.67, 0.68, 0.67, 0.65, 0.68, 0.67]
overlap25 =  [0.71, 0.73, 0.72, 0.73, 0.72, 0.73, 0.72, 0.71, 0.72, 0.72, 0.71, 0.70, 0.70, 0.71, 0.69, 0.69]
overlap50 =  [0.71, 0.74, 0.74, 0.74, 0.74, 0.75, 0.75, 0.75, 0.76, 0.74, 0.75, 0.74, 0.74, 0.74, 0.74, 0.74]
window_size= [  1,    3,    5,    7,    9,    11,   13,  15,  16,  17,   19,   21,   23,   25,  27,  29]


plt.plot(window_size, overlap0, 'o:', label='Overlap 0%')
plt.plot(window_size, overlap25, 'o:',label='Overlap 25%')
plt.plot(window_size, overlap50, 'o:', label='Overlap 50%')


# Add a vertical line at wsize = 7
plt.axvline(x=7, color='k', linestyle='--', label='$W_{TH}}$ Maximum window size')
plt.fill_between(window_size[:4], 0.81*np.ones(np.shape(window_size[:4])), np.zeros(np.shape(window_size[:4])), alpha=0.2)

plt.xlabel('Window size (sec)', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.xticks(window_size)  
plt.legend(loc='best', fontsize=10)

plt.ylim(0.64, 0.81)  # Adjust the limits as needed
plt.grid()
plt.savefig('window_size_effect_valence'+'.jpg', dpi=600)
plt.show()


#%%----------------------------------------------------------------
# Number of instances

# ML experiments planning - pc.xlsx (Authors' access only)


# sources
# c_sldl_1_2_1
# c_sldl_1_2_2
# c_sldl_1_2_3
# c_sldl_1_2_4
# c_sldl_1_2_5
# c_sldl_1_2_6
# c_sldl_1_2_7
# c_sldl_1_2_8
# c_sldl_1_2_9
# c_sldl_1_2_10
# c_sldl_1_2_11
# c_sldl_1_2_12
# c_sldl_1_2_13
# c_sldl_1_2_14
# c_sldl_1_2_15
# c_sldl_1_2_16
# c_sldl_1_2_17
# c_sldl_1_2_18
# c_sldl_1_2_19
# c_sldl_1_2_20
# c_sldl_1_2_21
# c_sldl_1_2_22
# c_sldl_1_2_56
# c_sldl_1_2_57
# c_sldl_1_2_58
# c_sldl_1_2_59
# c_sldl_1_2_60
# c_sldl_1_2_61



# Data
overlap0 =   [1276, 418, 251, 176, 137, 111, 93,  80,  76 , 70,  63,  55,  52, 45, 43, 40]
overlap25 =  [1681, 555, 333, 237, 182, 149, 123, 107, 100 , 94,  80,  73,  68, 61, 55, 53]
overlap50 =  [2548, 832, 497, 350, 270, 219, 184, 158, 147 , 137, 122, 108, 99, 88, 80, 76]
window_size = [ 1,   3,   5,   7,   9,   11, 13,  15, 16,   17,  19,  21,  23,  25,  27,  29]

# Define the width of each bar
bar_width = 0.2

# Define the x positions for each group of bars
x0 = np.arange(len(window_size))
x25 = x0 + bar_width
x50 = x25 + bar_width

# Create bar plots
plt.bar(x0, overlap0, width=bar_width, label='Overlap 0%')
plt.bar(x25, overlap25, width=bar_width, label='Overlap 25%')
plt.bar(x50, overlap50, width=bar_width, label='Overlap 50%')

plt.xlabel('Window size (sec)', fontsize=14)
plt.ylabel('Number of samples', fontsize=14)
# plt.title('Number of window instances for different window sizes')
plt.xticks(x0 + bar_width, window_size)  # Set x-axis ticks at the center of each group
plt.legend(loc='best', fontsize=14)
plt.grid()
# Get current axes
ax = plt.gca()
ax.set_yticks(np.arange(0, 2500, 200))  # Major ticks every 0.5 units on y-axis
# plt.ylim(0.625, 0.8)  # Adjust the limits as needed

plt.savefig('number_instances'+'.jpg', dpi=600)
# Show plot
plt.show()


#%%----------------------------------------------------------------
# # zoom:

# start_index = 4

# # Data
# window_size =  window_size[start_index:]
# x0 =  x0[start_index:]
# x25 =  x25[start_index:]
# x50 =  x50[start_index:]
# overlap0 =  overlap0[start_index:]
# overlap25 =  overlap25[start_index:]
# overlap50 =  overlap50[start_index:]


# plt.bar(x0, overlap0, width=bar_width, label='Overlap 0%')
# plt.bar(x25, overlap25, width=bar_width, label='Overlap 25%')
# plt.bar(x50, overlap50, width=bar_width, label='Overlap 50%')
# plt.xlabel('Window size (sec)', fontsize=14)
# plt.ylabel('Number of samples', fontsize=14)
# # plt.title('Number of window instances for different window sizes (Zoom on greater window sizes)')
# plt.xticks(x0 + bar_width, window_size)  # Set x-axis ticks at the center of each group
# plt.legend(loc='best', fontsize=14)
# plt.grid()
# # plt.ylim(0, 500)  # Adjust the limits as needed

# plt.savefig('number_instances_zoom'+'.jpg', dpi=600)
# # Show plot
# plt.show()


#%%----------------------------------------------------------------


# Strong, weak and classic labeling schemes (thresholds are far from 5)
# scheme = 'bipartite', number_thresholds = 2, threshold = 5, L = 3, H = 7

# ML experiments planning - pc.xlsx (summary of results - Authors' access only)

# sources: 
# c_sldl_1_2_23
# c_sldl_1_2_24
# c_sldl_1_2_25
# c_sldl_1_3_13
# c_sldl_1_3_14
# c_sldl_1_3_15
# c_sldl_1_3_16
# c_sldl_1_3_17
# c_sldl_1_3_18
# c_sldl_1_3_19
# c_sldl_1_3_20
# c_sldl_1_3_21
# c_sldl_1_3_22
# c_sldl_1_3_23
# c_sldl_1_3_24


# ------------------------ Arousal --------------------

# Data
# Wsize = 3 seconds

overlap0 =   [0.93, 0.84, 0.71]
overlap25 =  [0.93, 0.85, 0.73]
overlap50 =  [0.94, 0.86, 0.75]
window_size = ['Strong LS', 'Weak LS', 'Classical LS']

# Define the width of each bar
bar_width = 0.2

# Define the x positions for each group of bars
x0 = np.arange(len(window_size))
x25 = x0 + bar_width
x50 = x25 + bar_width

# Create bar plots
plt.bar(x0, overlap0, width=bar_width, label='Overlap 0%')
plt.bar(x25, overlap25, width=bar_width, label='Overlap 25%')
plt.bar(x50, overlap50, width=bar_width, label='Overlap 50%')

plt.xlabel('Labeling Scheme', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.xticks(x0 + bar_width, window_size)  # Set x-axis ticks at the center of each group
plt.legend(loc='best', fontsize=14)
plt.ylim(0.65, 0.95)  # Adjust the limits as needed

plt.savefig('labeling_scheme_arousal'+'.jpg', dpi=600)
plt.show()

# ------------- Valence ----------------------------------------------------------------

# Data
# Wsize = 3 seconds
overlap0 =   [0.91, 0.84, 0.71]
overlap25 =  [0.91, 0.85, 0.73]
overlap50 =  [0.92, 0.85, 0.74]
window_size = ['Strong LS', 'Weak LS', 'Classical LS']

# Define the width of each bar
bar_width = 0.2

# Define the x positions for each group of bars
x0 = np.arange(len(window_size))
x25 = x0 + bar_width
x50 = x25 + bar_width

# Create bar plots
plt.bar(x0, overlap0, width=bar_width, label='Overlap 0%')
plt.bar(x25, overlap25, width=bar_width, label='Overlap 25%')
plt.bar(x50, overlap50, width=bar_width, label='Overlap 50%')

plt.xlabel('Labeling Scheme', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.xticks(x0 + bar_width, window_size)  # Set x-axis ticks at the center of each group
plt.legend(loc='best', fontsize=14)
plt.ylim(0.65, 0.95)  # Adjust the limits as needed

plt.savefig('labeling_scheme_valence'+'.jpg', dpi=600)
# Show plot
plt.show()



#%%----------------------------------------------------------------
# Try Linear DEAP / K-EmoCon features in CASE ds.
# To check if ACC increases with more instances and if linear features can extract relevant information from the signal (keeping the same features).

# source:
# statistical features
# c_sldl_3_1_1 and derivatives
# c_sldl_3_1_1_bis
# c_sldl_3_1_2
# c_sldl_3_1_3
# c_sldl_3_1_4
# c_sldl_3_1_5
# c_sldl_3_1_6
# c_sldl_3_1_7
# c_sldl_3_1_8
# c_sldl_3_1_9
# c_sldl_3_1_16
# c_sldl_3_1_17
# c_sldl_3_1_18
# nonlinear features
# c_sldl_1_2_1
# c_sldl_1_1
# c_sldl_1_2
# c_sldl_1_2_2
# c_sldl_1_2_3
# c_sldl_1_2_4
# c_sldl_1_2_5
# c_sldl_1_2_6
# c_sldl_1_2_7
# c_sldl_1_2_11
# c_sldl_1_2_12
# c_sldl_1_2_13
# c_sldl_1_2_14
# c_sldl_1_2_15
# c_sldl_1_2_16


# Data 
# list Columns : window size: 3, 5, 7, 8, 11, 16 sec, respectively. 

val_overlap0_acc_st =   [0.65, 0.65, 0.66,   0.65, 0.65, 0.65]  # st _ statistical features
val_overlap0_uar_st =   [0.59, 0.60, 0.61,   0.61, 0.61, 0.61]  
aro_overlap0_acc_st =   [0.64, 0.65, 0.65,   0.65, 0.65, 0.64]  
aro_overlap0_uar_st =   [0.61, 0.62, 0.62,   0.62, 0.62, 0.62]  

val_overlap0_acc_nl =   [0.71, 0.71,  0.71,  0.70, 0.70, 0.68]  # nl _ non linear features
val_overlap0_uar_nl =   [0.64, 0.68,  0.67,  0.67, 0.66, 0.65]  
aro_overlap0_acc_nl =   [0.71, 0.71,  0.71,  0.70, 0.70, 0.68] 
aro_overlap0_uar_nl =   [0.66, 0.68,  0.68,  0.68, 0.67, 0.65]  

val_overlap25_acc_st =   [0.67, 0.67, 0.66,  0.66, 0.67, 0.67] 
val_overlap25_uar_st =   [0.62, 0.62, 0.62,  0.62, 0.62, 0.63]
aro_overlap25_acc_st =   [0.68, 0.68, 0.67,  0.66, 0.67, 0.67]  
aro_overlap25_uar_st =   [0.65, 0.64, 0.65,  0.63, 0.64, 0.65]  

val_overlap25_acc_nl =   [0.73, 0.72, 0.73,  0.73, 0.73, 0.72] 
val_overlap25_uar_nl =   [0.70, 0.69, 0.70,  0.69, 0.69, 0.69]  
aro_overlap25_acc_nl =   [0.72, 0.73, 0.72,  0.72, 0.73, 0.71] 
aro_overlap25_uar_nl =   [0.70, 0.7,  0.70,  0.70, 0.70, 0.69]  

val_overlap50_acc_st =   [0.68, 0.68, 0.69,  0.69, 0.69, 0.71] 
val_overlap50_uar_st =   [0.62, 0.63, 0.64,  0.65, 0.65, 0.67]
aro_overlap50_acc_st =   [0.69, 0.68, 0.69,  0.69, 0.71, 0.69]  
aro_overlap50_uar_st =   [0.64, 0.66, 0.67,  0.67, 0.69, 0.66]  

val_overlap50_acc_nl =   [0.74, 0.74, 0.74,  0.75, 0.75, 0.76]  # c_sldl_1_2, c_sldl_1_2_4, c_sldl_1_2_7 
val_overlap50_uar_nl =   [0.71, 0.71, 0.71,  0.72, 0.72, 0.73]  
aro_overlap50_acc_nl =   [0.75, 0.75, 0.75,  0.76, 0.76, 0.75] 
aro_overlap50_uar_nl =   [0.73, 0.73, 0.73,  0.74, 0.74, 0.73]  

# 3 seconds window size
val_acc_st = [val_overlap0_acc_st[0], val_overlap25_acc_st[0] ,val_overlap50_acc_st[0]]
val_acc_nl = [val_overlap0_acc_nl[0], val_overlap25_acc_nl[0] ,val_overlap50_acc_nl[0]]

aro_acc_st = [aro_overlap0_acc_st[0], aro_overlap25_acc_st[0] ,aro_overlap50_acc_st[0]]
aro_acc_nl = [aro_overlap0_acc_nl[0], aro_overlap25_acc_nl[0] ,aro_overlap50_acc_nl[0]]

overlap = [0,0.25, 0.5]

#%%----------------------------------------------------------------

# 3 seconds window size
wi = 0 # window size index
val_acc_st = [val_overlap0_acc_st[wi], val_overlap25_acc_st[wi] ,val_overlap50_acc_st[wi]]
val_acc_nl = [val_overlap0_acc_nl[wi], val_overlap25_acc_nl[wi] ,val_overlap50_acc_nl[wi]]

aro_acc_st = [aro_overlap0_acc_st[wi], aro_overlap25_acc_st[wi] ,aro_overlap50_acc_st[wi]]
aro_acc_nl = [aro_overlap0_acc_nl[wi], aro_overlap25_acc_nl[wi] ,aro_overlap50_acc_nl[wi]]


overlap = [0,0.25, 0.5]

x0 = np.arange(len(overlap))
x25 = x0 + bar_width

# Define the width of each bar
bar_width = 0.2

# Create bar plots
plt.bar(x0, aro_acc_st, width=bar_width, label='Statistical feat.')
plt.bar(x25, aro_acc_nl, width=bar_width, label='Nonlinear')

plt.xlabel('Overlap', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
# plt.title('Arousal accuracy for window size = 11 sec  | Random Forest')
plt.xticks(x0 + bar_width, overlap)  # Set x-axis ticks at the center of each group
plt.legend(loc='best', fontsize=14)
plt.ylim(0.6, 0.77)  # Adjust the limits as needed

plt.savefig('stats_vs_non_linear_3sw_arousal'+'.jpg', dpi=600)
# Show plot
plt.show()


#%%----------------------------------------------------------------
# 3 seconds window size
wi = 0 # window size index
val_acc_st = [val_overlap0_acc_st[wi], val_overlap25_acc_st[wi] ,val_overlap50_acc_st[wi]]
val_acc_nl = [val_overlap0_acc_nl[wi], val_overlap25_acc_nl[wi] ,val_overlap50_acc_nl[wi]]

aro_acc_st = [aro_overlap0_acc_st[wi], aro_overlap25_acc_st[wi] ,aro_overlap50_acc_st[wi]]
aro_acc_nl = [aro_overlap0_acc_nl[wi], aro_overlap25_acc_nl[wi] ,aro_overlap50_acc_nl[wi]]


overlap = [0,0.25, 0.5]

x0 = np.arange(len(overlap))
x25 = x0 + bar_width

# Define the width of each bar
bar_width = 0.2

# Create bar plots
plt.bar(x0, val_acc_st, width=bar_width, label='Statistical feat.')
plt.bar(x25, val_acc_nl, width=bar_width, label='Nonlinear')

plt.xlabel('Overlap', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.xticks(x0 + bar_width, overlap)  # Set x-axis ticks at the center of each group
plt.legend(loc='best', fontsize=14)
plt.ylim(0.6, 0.77)  # Adjust the limits as needed

plt.savefig('stats_vs_non_linear_3sw_valence'+'.jpg', dpi=600)
# Show plot
plt.show()

#%%---------------------------------------------------------------------


#%%----------------------------------------------------------------
# plot participant annotation within a window (with overlap and without overlap)

import numpy as np
import random as python_random
import socket
from matplotlib.ticker import FuncFormatter  # Import FuncFormatter

# ---------------  import constants and functions:
# https://stackoverflow.com/questions/15514593/importerror-no-module-named-when-trying-to-run-python-script
import sys, os

# https://stackoverflow.com/questions/8663076/python-best-way-to-add-to-sys-path-relative-to-the-current-running-script
from definitions import *
from functions import * #proyect's functions / defs

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
except NameError:
    # assign a participant in order to run the whole script:
    participant = str(28)  # original = 28  u 11
        

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
# window_seconds = 25
# overlap_percentage = 0.0
# combined_df, combined_df_median = get_dataframes(window_seconds, overlap_percentage)

# index = 0
# num_points = len(combined_df["aro_chunks"].iloc[index])
# time_ms = range(num_points) # A simple range from 0 to num_points - 1

def plot_label_within_window(window_seconds, overlap_percentage):

    combined_df, combined_df_median = get_dataframes(window_seconds, overlap_percentage)

    # Convert the number to a string
    num_str = str(overlap_percentage)
    # Split the string at the decimal point
    integer_part, decimal_part = num_str.split(".")

    
    # Function to format y-axis labels with commas
    def format_with_commas(x, pos):
        return '{:,.0f}'.format(x)


    index = 0
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_with_commas))
    plt.plot(combined_df["aro_chunks"].iloc[index], label='Window 1')
    plt.plot(combined_df_median["aro_chunks"].iloc[index], label='Median Window 1')
    plt.ylabel('Label', fontsize=14)
    plt.xlabel('Time (ms)', fontsize=14)
    plt.legend(loc='best',fontsize=10)

    index = 1
    plt.plot(combined_df["aro_chunks"].iloc[index], label='Window 2')
    plt.plot(combined_df_median["aro_chunks"].iloc[index], label='Median Window 2')
    plt.ylabel('Label',fontsize=14)
    plt.xlabel('Time (ms)',fontsize=14)
    plt.legend(loc='best',fontsize=10)
    plt.savefig('Overlap_Rein_part_28_ws_25_ov_'+ decimal_part + '.jpg', dpi=1000)
    # plt.savefig('Overlap_Rein_part_28_ws_25_ov_025'+'.jpg', dpi=1000)
    plt.show()
    print(str(decimal_part))

    return combined_df, combined_df_median


window_seconds = 25
overlap_percentage = 0.0
combined_df, combined_df_median = plot_label_within_window(window_seconds, overlap_percentage)

window_seconds = 25
overlap_percentage = 0.25
combined_df, combined_df_median = plot_label_within_window(window_seconds, overlap_percentage)



#%%----------------------------------------------------------------
# plot a participant annotation (with windowing)

num_of_windows = combined_df.shape[0]
precision = 2
prevalence_w_aro = 0
prevalence_sum_aro = 0
prevalence_w_val = 0
prevalence_sum_val = 0

for index in range(num_of_windows):
    plt.plot(combined_df["val_chunks"].iloc[index], label='Window 1')
    plt.plot(combined_df_median["val_chunks"].iloc[index], label='Median Window 1')
    plt.ylabel('Label')
    plt.xlabel('Time (ms)')
    

# plt.savefig('participant_labels_windowing'+'.jpg', dpi=1000)
plt.show(block=True)

#%%----------------------------------------------------------------




#%%----------------------------------------------------------------
# Ground Truth label Dominance study - results

from numpy import loadtxt
import matplotlib.pyplot as plt
import numpy as np

prevalence_over_val = loadtxt('prevalence_over_val.csv', delimiter=',')
prevalence_over_aro = loadtxt('prevalence_over_aro.csv', delimiter=',')

prevalence_over_aro_std = loadtxt('prevalence_over_aro_std.csv', delimiter=',')
prevalence_over_val_std = loadtxt('prevalence_over_val_std.csv', delimiter=',')


#%%----------------------------------------------------------------

# ------------------------------ Arousal -------------------------------------

overlap0 =   prevalence_over_aro[0,:]/100
overlap25 =  prevalence_over_aro[1,:]/100
overlap50 =  prevalence_over_aro[2,:]/100
window_size= [  1,    3,    5,    7,    9,    11,   13,  15, 16, 17,   19,   21,   23,   25,   27,  29]

plt.plot(window_size, overlap0, 'o:', label='Overlap 0%')
plt.plot(window_size, overlap25, 'o:',label='Overlap 25%')
plt.plot(window_size, overlap50, 'o:', label='Overlap 50%')

plt.xlabel('Window size (sec)', fontsize=14)
plt.ylabel('Dominance', fontsize=14)
plt.xticks(window_size)
# plt.title('Dominance - Ground truth label for different window sizes (Arousal labels)')
plt.legend(loc='best', fontsize=14)
# plt.ylim(0.625, 0.8)  # Adjust the limits as needed
plt.grid()
# plt.savefig('dominance_arousal'+'.jpg', dpi=600)
plt.show()



# ------------------------------ Arousal standard deviation -------------------------------------

overlap0 =   prevalence_over_val_std[0,:]/100
overlap25 =  prevalence_over_val_std[1,:]/100
overlap50 =  prevalence_over_aro_std[2,:]/100
# overlap50 =  prevalence_wsize_aro_std/100
window_size= [  1,    3,    5,    7,    9,    11,   13,  15, 16, 17,   19,   21,   23,   25,   27,  29]

plt.plot(window_size, overlap0, 'o:', label='Overlap 0%')
plt.plot(window_size, overlap25, 'o:',label='Overlap 25%')
plt.plot(window_size, overlap50, 'o:', label='Overlap 50%')

plt.xlabel('Window size (sec)', fontsize=14)
plt.ylabel('Dominance', fontsize=14)
plt.xticks(window_size)
# plt.title('Dominance standard deviation - Ground truth label for different window sizes (Arousal labels)')
plt.legend(loc='best', fontsize=14)
# plt.ylim(0.625, 0.8)  # Adjust the limits as needed
plt.grid()
# plt.savefig('dominance_arousal_std'+'.jpg', dpi=600)
plt.show()


#%%----------------------------------------------------------------

# ------------------------------ Dominance with Error Bands (Arousal) -------------------------------------

# Mean dominance values
dominance_mean_0 = prevalence_over_aro[0,:] / 100
dominance_mean_25 = prevalence_over_aro[1,:] / 100
dominance_mean_50 = prevalence_over_aro[2,:] / 100

# Standard deviation values
dominance_std_0 = prevalence_over_aro_std[0,:] / 100
dominance_std_25 = prevalence_over_aro_std[1,:] / 100
dominance_std_50 = prevalence_over_aro_std[2,:] / 100




# Window sizes
window_size = [1, 3, 5, 7, 9, 11, 13, 15, 16, 17, 19, 21, 23, 25, 27, 29]



# Plot mean dominance with error bands
plt.plot(window_size, dominance_mean_0, 'o-', label='Overlap 0%')

plt.fill_between(window_size, 
                 dominance_mean_0 - dominance_std_0, 
                 dominance_mean_0 + dominance_std_0, 
                 alpha=0.2)

plt.plot(window_size, dominance_mean_25, 'o-', label='Overlap 25%')
plt.fill_between(window_size, 
                 dominance_mean_25 - dominance_std_25, 
                 dominance_mean_25 + dominance_std_25, 
                 alpha=0.2)

plt.plot(window_size, dominance_mean_50, 'o-', label='Overlap 50%')
plt.fill_between(window_size, 
                 dominance_mean_50 - dominance_std_50, 
                 dominance_mean_50 + dominance_std_50, 
                 alpha=0.2)

# dominance ground truth (GT) label's minimum threshold
# $D_{GT_{\text{min}}
GT_min_threshold_value = 0.5
GT_min_threshold = GT_min_threshold_value*np.ones(np.shape(window_size))
plt.plot(window_size, GT_min_threshold, '--', label='Minimum $D_{GT}$ Threshold')


# Add a vertical line at wsize = 3
plt.axvline(x=3, color='k', linestyle='--', label='$W_{TH}}$ Maximum window size')
y_range = plt.gca().get_ylim()
plt.fill_between(window_size[:2], 0.9*np.ones(np.shape(window_size[:2])), np.zeros(np.shape(window_size[:2])), alpha=0.2)


plt.xlabel('Window size (sec)', fontsize=14)
plt.ylabel('Dominance', fontsize=14)
plt.xticks(window_size)
plt.legend(loc='best', fontsize=12)
plt.grid()
plt.ylim(0.18, 0.9)  # Adjust the limits as needed
plt.savefig('dominance_with_error_bands_arousal.jpg', dpi=1000)
plt.show()



#%%----------------------------------------------------------------
# --------------------------------  Valence -----------------------------------------

# Data

overlap0 =   prevalence_over_val[0,:]/100
overlap25 =  prevalence_over_val[1,:]/100
overlap50 =  prevalence_over_val[2,:]/100
window_size= [  1,    3,    5,    7,    9,    11,   13,  15,  16,  17,   19,   21,   23,   25,  27,  29]


plt.plot(window_size, overlap0, 'o:', label='Overlap 0%')
plt.plot(window_size, overlap25, 'o:',label='Overlap 25%')
plt.plot(window_size, overlap50, 'o:', label='Overlap 50%')


plt.xlabel('Window size (sec)', fontsize=14)
plt.ylabel('Dominance', fontsize=14)
plt.xticks(window_size)  
# plt.title('Dominance - Ground truth label for different window sizes (Valence labels)')
plt.legend(loc='best', fontsize=14)
# plt.ylim(0.625, 0.8)  # Adjust the limits as needed
plt.grid()
# plt.savefig('dominance_valence'+'.jpg', dpi=600)
plt.show()


# ------------------------------ Valence standard deviation -------------------------------------

overlap0 =   prevalence_over_val_std[0,:]/100
overlap25 =  prevalence_over_val_std[1,:]/100
overlap50 =  prevalence_over_val_std[2,:]/100
window_size= [  1,    3,    5,    7,    9,    11,   13,  15, 16, 17,   19,   21,   23,   25,   27,  29]

plt.plot(window_size, overlap0, 'o:', label='Overlap 0%')
plt.plot(window_size, overlap25, 'o:',label='Overlap 25%')
plt.plot(window_size, overlap50, 'o:', label='Overlap 50%')

plt.xlabel('Window size (sec)', fontsize=14)
plt.ylabel('Dominance', fontsize=14)
plt.xticks(window_size)
# plt.title('Dominance standard deviation - Ground truth label for different window sizes (Valence labels)')
plt.legend(loc='best', fontsize=14)
# plt.ylim(0.625, 0.8)  # Adjust the limits as needed
plt.grid()
# plt.savefig('dominance_valence_std'+'.jpg', dpi=600)
plt.show()



#%%----------------------------------------------------------------

# ------------------------------ Dominance with Error Bands (Valence) -------------------------------------

# Mean dominance values
dominance_mean_0 = prevalence_over_val[0,:] / 100
dominance_mean_25 = prevalence_over_val[1,:] / 100
dominance_mean_50 = prevalence_over_val[2,:] / 100

# Standard deviation values
dominance_std_0 = prevalence_over_val_std[0,:] / 100
dominance_std_25 = prevalence_over_val_std[1,:] / 100
dominance_std_50 = prevalence_over_val_std[2,:] / 100

# Window sizes
window_size = [1, 3, 5, 7, 9, 11, 13, 15, 16, 17, 19, 21, 23, 25, 27, 29]

# Plot mean dominance with error bands
plt.plot(window_size, dominance_mean_0, 'o-', label='Overlap 0%')
plt.fill_between(window_size, 
                 dominance_mean_0 - dominance_std_0, 
                 dominance_mean_0 + dominance_std_0, 
                 alpha=0.2)

plt.plot(window_size, dominance_mean_25, 'o-', label='Overlap 25%')
plt.fill_between(window_size, 
                 dominance_mean_25 - dominance_std_25, 
                 dominance_mean_25 + dominance_std_25, 
                 alpha=0.2)

plt.plot(window_size, dominance_mean_50, 'o-', label='Overlap 50%')
plt.fill_between(window_size, 
                 dominance_mean_50 - dominance_std_50, 
                 dominance_mean_50 + dominance_std_50, 
                 alpha=0.2)

# dominance ground truth (GT) label's minimum threshold
# $D_{GT_{\text{min}}
GT_min_threshold_value = 0.5
GT_min_threshold = GT_min_threshold_value*np.ones(np.shape(window_size))
plt.plot(window_size, GT_min_threshold, '--', label='Minimum $D_{GT}$ Threshold')

# Add a vertical line at wsize = 7
plt.axvline(x=7, color='k', linestyle='--', label='$W_{TH}}$ Maximum window size')
plt.fill_between(window_size[:4], 1*np.ones(np.shape(window_size[:4])), np.zeros(np.shape(window_size[:4])), alpha=0.2)


plt.xlabel('Window size (sec)', fontsize=14)
plt.ylabel('Dominance', fontsize=14)
plt.xticks(window_size)
plt.legend(loc='best', fontsize=12)
plt.grid()
plt.ylim(0.26, 1)  # Adjust the limits as needed

plt.savefig('dominance_with_error_bands_valence.jpg', dpi=1000)
plt.show()

#%%----------------------------------------------------------------



#%%----------------------------------------------------------------
# Segment labels mapping: from continuous multivalued labels to one value annotation per segment.

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


#%%----------------------------------------------------------------
# Show timestamps in the figure

NEUTRAL_SCREEN = 1
STIMULUS_SCREEN = 2

def map_video_to_tag(video_id):
    """
    map Video-ID to stimulus tag
    Video-ID  >= 10 will be assigned stimulus tag = 1, corresponding to the blue screen/initial screen.
    Video-ID < 10 will be assigned stimulus tag = 2, corresponding to a true stimulus.
    """
    if video_id >= 10:
        stimulus_tag = NEUTRAL_SCREEN
    elif video_id < 10:
        stimulus_tag = STIMULUS_SCREEN
    return stimulus_tag


stimulus_tag_list = [map_video_to_tag(i) for i in data['video']] 

data['tag'] = stimulus_tag_list  # dataframe now contains the stimulus tag of each video


def scale_tag(tag,min_value,max_value):
    """
    Scales a two valued vector (Low/high) to the min / max value of a signal.
    It's useful to visualize the stimulus timestamps in the physiological signal altogether 

    """ 
    if tag == NEUTRAL_SCREEN:
        scaled_tag = min_value
    elif tag == STIMULUS_SCREEN:
        scaled_tag = max_value
    return scaled_tag

#%%----------------------------------------------------------------
# Mean-centered normalization
# Zitouni et al (2023)

# calculate mean of the baseline (the last 60 sec of starting video. CASE has a 101 second baseline)

START_VIDEO_DURATION = 101500   # ms 
BASELINE_WINDOW = 60000 # ms 

ppg_baseline_mean = data['bvp'][START_VIDEO_DURATION - BASELINE_WINDOW : START_VIDEO_DURATION].mean()
gsr_baseline_mean = data['gsr'][START_VIDEO_DURATION - BASELINE_WINDOW : START_VIDEO_DURATION].mean()

ppg_mc = data['bvp'] - ppg_baseline_mean  # mean centered based normalization
gsr_mc = data['gsr'] - gsr_baseline_mean # mean centered based normalization

data_mc = data.copy()  # mean centered copy
data_mc['bvp'] = ppg_mc
data_mc['gsr'] = gsr_mc



#%%----------------------------------------------------------------
# Extract the cleaned GSR signal

# Process the raw EDA signal
gsr_signals, gsr_info = nk.eda_process(data_mc['gsr'], sampling_rate=SAMPLING_RATE, method="neurokit")


#%%----------------------------------------------------------------
# Extract the cleaned PPG signal from the mean-centered normalized signal

# Process the raw PPG signal
ppg_signals, ppg_info = nk.ppg_process(data_mc['bvp'], sampling_rate=SAMPLING_RATE)



#%% ----------------------------------------------------------------
# # GSR + PPG signal with stimulus timestamps

gsr_min_value = min(data['gsr'])
gsr_max_value = max(data['gsr'])
gsr_scaled_tag = [scale_tag(i,gsr_min_value,gsr_max_value) for i in data['tag']] 

min_value = min(data['bvp'])
max_value = max(data['bvp'])
ppg_scaled_tag = [scale_tag(i,min_value,max_value) for i in data['tag']] 


#%%----------------------------------------------------------------
# Windowing / segmentation

# modified version of https://chat.openai.com/c/222cb1dc-b798-46a4-ba3e-dd076db90a71

import pandas as pd

def standarize_signal(signal):

        mean = np.mean(signal)   # mean value
        std = np.std(signal)     # standard deviation

        if np.isclose(std, 0.0):
            print("std is close to zero: ", std)

        standarized_signal = [(x-mean)/std for x in signal]

        return standarized_signal


def segment_time_series(time_series, window_size, overlap=False, stride=1, standardize = 0):
    """
    Perform windowing on a time-series.
    The window size is set to window_size.
    If overlap is set to true, the sliding window step is equal to the stride; otherwise, the step is set to window_size.
    In the former case, the percentage of overlap is calculated as [(stride / window size) * 100].
    If the length of the time_series is not a multiple of the step, the last segment will be lost and not processed.
    """
    segments = []
    step = stride if overlap else window_size
    # index = []
    for i in range(0, len(time_series) - window_size + 1, step):
        segment = time_series[i:i + window_size]
        if standardize == 1:
            segment = standarize_signal(segment)    
        segments.append(segment)
        # index.append(i)

    # return segments, index
    return segments

# ppg signal windowing
time_series_data = ppg_signals['PPG_Clean'].copy()

# test windowing
window_size = 3*1000  # ms
step = int(2.25*1000) #ms
overlap = True



#%%----------------------------------------------------------------

# Process:
# 1. Baseline mean normalization (subtract A from all the time series).
# 2. Perform segmentation on a B-windows basis (the first chunk/window start should be synced with the B-window start).
# 3. Discard the last chunk (inside a B-window) if window_size/step doesn’t fit exactly with B-window duration.
# 4. Discard all data corresponding to C-windows.


valence_ups = nk.signal_resample(valence, sampling_rate=20, desired_sampling_rate=1000, method="numpy")  # ups = upsampled
valence_ups = pd.Series(valence_ups)  #annotation must be a pandas series, because segment_time_series expects data in this format
arousal_ups = nk.signal_resample(arousal, sampling_rate=20, desired_sampling_rate=1000, method="numpy")  # ups = upsampled
arousal_ups = pd.Series(arousal_ups)  #annotation must be a pandas series, because segment_time_series expects data in this format

data_cleaned = data_mc.copy()  # ppg and gsr filtered copy
data_cleaned['bvp'] = ppg_signals['PPG_Clean']
data_cleaned['gsr'] = gsr_signals['EDA_Clean']
data_cleaned['val'] = valence_ups
data_cleaned['aro'] = arousal_ups

# Identify rows containing neutral screens (start video, end video, blue screen)
mask_VIDEO_ID_10 = (seqs_order_num == VIDEO_ID_10).any(axis=1)
mask_VIDEO_ID_11 = (seqs_order_num == VIDEO_ID_11).any(axis=1)
mask_VIDEO_ID_12 = (seqs_order_num == VIDEO_ID_12).any(axis=1)

mask = mask_VIDEO_ID_10 | mask_VIDEO_ID_11 | mask_VIDEO_ID_12 # contains all neutral screens

filtered_seqs_order_num = seqs_order_num[~mask]  #contains stimulii sequence order (without neutral screens)

# total_num_of_segments = 0

# def combine_chuncks(df_base, segments_pd, video_id, first_time = True):
def combine_chuncks(df_base, segments_pd, video_id, time_series_name):
    """
    Combine chuncks in order, taking into account all the stimulus windows in order.
    
    segments_pd = pandas Series
    video_id = constant number
    if add_video = 1, it adds a column to the pd.Series with the video_id number. 
    time_series_name is a string with the time series name. 
    """

    # Create a new Pandas Series with the video-ID (constant number)
    video_column = pd.Series( [video_id] * len(segments_pd))

    # Concatenate the original Series and the new Series along the columns
    combined_df = pd.concat([segments_pd, video_column], axis=COLUMNS)

    # Rename the columns:
    combined_df.columns = [time_series_name + '_' +'chuncks', 'video']
    # print("header combined_df: ", list(combined_df.columns))

    # if add_video == 0:
    #     combined_df.drop(labels = 'video', axis = 1)
    #     print("header combined_df: ", list(combined_df.columns))

    # build a dataFrame with all chunks of all the stimulus windows together
    if df_base.equals(segments_pd):
        df = combined_df.copy()
    else:
        df = pd.concat([df_base, combined_df], axis = INDEX)
    
    # Display the resulting DataFrame
    # print(df)

    return df

# loop over all 8 stimulii (each stimulus is then windowed)
for i in range(filtered_seqs_order_num.shape[0]):
# for i in range(1):
    # one particular stimulus window with all its data (physiological, val, aro, etc):
    stimulus_window = data_cleaned[data_cleaned['video'] == filtered_seqs_order_num['sub_' + participant].iloc[i]]    # contains one particular B-Window

    gsr_segments = segment_time_series(stimulus_window['gsr'], window_size, overlap,step, standardize = 0)  # gsr_segments for one particular stimulus window
    ppg_segments = segment_time_series(stimulus_window['bvp'], window_size, overlap,step, standardize = 0)
    val_segments = segment_time_series(stimulus_window['val'], window_size, overlap,step, standardize = 0)
    aro_segments = segment_time_series(stimulus_window['aro'], window_size, overlap,step, standardize = 0)
    time_vector_segments = segment_time_series(stimulus_window['daqtime'], window_size, overlap,step, standardize = 0)
    tag_segments = segment_time_series(stimulus_window['tag'], window_size, overlap,step, standardize = 0)


    gsr_segments_pd = pd.Series(gsr_segments)
    ppg_segments_pd = pd.Series(ppg_segments)
    val_segments_pd = pd.Series(val_segments)
    aro_segments_pd = pd.Series(aro_segments)
    time_vector_segments_pd = pd.Series(time_vector_segments)
    tag_segments_pd = pd.Series(tag_segments)

    # build a dataFrame with all chunks of all the stimulus windows together (one dataFrame for each time series)
    if i == 0:

        # gsr_df = combine_chuncks(gsr_segments_pd, gsr_segments_pd, filtered_seqs_order_num['sub_' + participant].iloc[i])
        gsr_df = combine_chuncks(gsr_segments_pd, gsr_segments_pd, filtered_seqs_order_num['sub_' + participant].iloc[i], 'gsr')
        ppg_df = combine_chuncks(ppg_segments_pd, ppg_segments_pd, filtered_seqs_order_num['sub_' + participant].iloc[i], 'ppg')
        val_df = combine_chuncks(val_segments_pd, val_segments_pd, filtered_seqs_order_num['sub_' + participant].iloc[i], 'val')
        aro_df = combine_chuncks(aro_segments_pd, aro_segments_pd, filtered_seqs_order_num['sub_' + participant].iloc[i], 'aro')
        time_df = combine_chuncks(time_vector_segments_pd, time_vector_segments_pd, filtered_seqs_order_num['sub_' + participant].iloc[i], 'time')
        tag_df = combine_chuncks(tag_segments_pd, tag_segments_pd, filtered_seqs_order_num['sub_' + participant].iloc[i], 'tag')
    else:
        # gsr_df = combine_chuncks(gsr_df, gsr_segments_pd, filtered_seqs_order_num['sub_' + participant].iloc[i])
        gsr_df = combine_chuncks(gsr_df, gsr_segments_pd, filtered_seqs_order_num['sub_' + participant].iloc[i], 'gsr')
        ppg_df = combine_chuncks(ppg_df, ppg_segments_pd, filtered_seqs_order_num['sub_' + participant].iloc[i], 'ppg')
        val_df = combine_chuncks(val_df, val_segments_pd, filtered_seqs_order_num['sub_' + participant].iloc[i], 'val')
        aro_df = combine_chuncks(aro_df, aro_segments_pd, filtered_seqs_order_num['sub_' + participant].iloc[i], 'aro')
        time_df = combine_chuncks(time_df, time_vector_segments_pd, filtered_seqs_order_num['sub_' + participant].iloc[i], 'time')
        tag_df = combine_chuncks(tag_df, tag_segments_pd, filtered_seqs_order_num['sub_' + participant].iloc[i], 'tag')


combined_df = pd.concat([gsr_df, ppg_df, val_df, aro_df, time_df, tag_df], axis=COLUMNS)


# Remove duplicated 'video' columns except one
current_columns = list(combined_df.columns)
current_columns[1] = 'video_id'
combined_df.columns = current_columns
list(combined_df.columns)

# Find duplicate column names
duplicate_columns = combined_df.columns[combined_df.columns.duplicated()]

# Drop columns with duplicate names
combined_df = combined_df.drop(columns=duplicate_columns)
list(combined_df.columns)

# Move the 'video' column to the end
video_column = combined_df.pop('video_id')
combined_df['video'] = video_column
list(combined_df.columns)

#%%----------------------------------------------------------------
# check correct df combination:

# lines should fully overlap, if combination was correct
plt.plot(gsr_df['gsr_chuncks'].iloc[0], label="from original chunck")
plt.plot(combined_df['gsr_chuncks'].iloc[0], label="from combined")

# lines should fully overlap, if combination was correct
plt.plot(gsr_df['gsr_chuncks'].iloc[1], label="from original chunck")
plt.plot(combined_df['gsr_chuncks'].iloc[1], label="from combined")

# lines should fully overlap, if combination was correct
plt.plot(gsr_df['gsr_chuncks'].iloc[2], label="from original chunck")
plt.plot(combined_df['gsr_chuncks'].iloc[2], label="from combined")



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


print("-------------------------------")
chunck_number = 12
plt.plot(val_segments[chunck_number][:], label="label segment")
plt.legend()
plt.show(block=True)

# test voting ALG with physiological data:
print("test voting ALG with physiological data:")
n = len((val_segments[chunck_number][:]))
majority = findMajority(list(val_segments[chunck_number][:]), n)
print(" The majority element is :" ,majority)


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
        
        plt.legend(fontsize=14)
        plt.ylabel("Annotation", fontsize=14)
        plt.xlabel("Time (seconds)", fontsize=14)
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
compare_annotation_chunck_decision_methods(val_segments, method = "all", chunck_number = 27)


#%%----------------------------------------------------------------
# labels class distribution (number of points)

#%% --------------------------------------------------
from points_per_classf import get_dataframes_one_participant
import numpy as np
import pandas as pd
from numpy import savetxt
from numpy import loadtxt
from sys import platform
# from openpyxl import load_workbook
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter  # Import FuncFormatter
import neurokit2 as nk # Load the NeuroKit package

participants = np.arange(1, 31)  # participant from 1 to 30

with tf.device('/CPU:0'):

    # Record the start time
    start_time = time.time()
    
    window_seconds = 3  # can be the minimum (to avoid lossing lables due to wsize not multiple of B-window)
    overlap_percentage = 0.0

    for participant_index in participants:

        if participant_index == 1:
            combined_df, combined_df_median = get_dataframes_one_participant(participant_index,window_seconds, overlap_percentage)
            combined_series = combined_df.copy()
        else:
            combined_df, combined_df_median = get_dataframes_one_participant(participant_index,window_seconds, overlap_percentage) 
            combined_series = pd.concat([combined_series, combined_df], ignore_index=True)
        # print("combined_series.shape: ", combined_series.shape)
    
    #%%----------------------------------------------------------------
    val_chunks_flattened = combined_series["val_chunks"].explode().reset_index(drop=True)
    aro_chunks_flattened = combined_series["aro_chunks"].explode().reset_index(drop=True)
    # val_chunks_flattened.to_csv('labels_vh.csv', index=False)


    #%%----------------------------------------------------------------

    def rescale(series, old_min, old_max, new_min, new_max):
        return ((series - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

    # Applying rescaling to the Series
    val_chunks_flattened_res = rescale(val_chunks_flattened, old_min=0.5, old_max=9.5, new_min=1, new_max=9)
    aro_chunks_flattened_res = rescale(aro_chunks_flattened, old_min=0.5, old_max=9.5, new_min=1, new_max=9)

    #%%----------------------------------------------------------------
    # Downsample to 20 Hz (original annotation freq. )

    def upsample_annotations(annotation, _sampling_rate=20, _desired_sampling_rate = 1000, _method = "numpy", show_plot=1):
        """
        Upsample labels / annotations time_series (to have the same sample rate as biosignals)
        Verification of correct annotations upsampling

        annotation: valence or arousal annotation
        sampling_rate and desired_sampling_rate in Hz. 
        method = "numpy" (default). Others: ["interpolation", "FFT", "poly", "pandas"]

        returns: upsampled annotation time-serie.
        """

        annotation = np.array(annotation, dtype=np.float64)
        annotation_upsampled = nk.signal_resample(annotation, sampling_rate=_sampling_rate, desired_sampling_rate=_desired_sampling_rate, method=_method)


        if show_plot:
            # numpy preserves original waveform:
            plt.figure()
            plt.plot(annotation_upsampled,label='numpy')
            plt.plot(annotation, label='original')
            plt.legend()

            plt.figure()
            plt.plot(annotation_upsampled,label='numpy')
            plt.legend()
            plt.figure()
            plt.plot(annotation, label='original')
            plt.legend()
            plt.show(block=True)

        return annotation_upsampled

    # check:
    val_chunks_flattened_res_ds = upsample_annotations(val_chunks_flattened_res, _sampling_rate=1000, _desired_sampling_rate = 20, _method = "numpy", show_plot=0) # ds = downsampled
    aro_chunks_flattened_res_ds = upsample_annotations(aro_chunks_flattened_res, _sampling_rate=1000, _desired_sampling_rate = 20, _method = "numpy", show_plot=0) # ds = downsampled
    plt.figure()
    plt.plot(val_chunks_flattened_res,label='1000 Hz')
    plt.legend()
    plt.figure()
    plt.plot(val_chunks_flattened_res_ds, label='20 Hz')
    plt.legend()
    plt.show(block=True)


    #%%--------------------------------------
    plt.figure()
    plt.plot(val_chunks_flattened_res_ds)
    plt.title('Valence')
    plt.savefig('aro_chunks_flattened_res.jpg', dpi=1000)
    plt.show(block=True)
    plt.figure()
    plt.plot(val_chunks_flattened_res_ds)
    plt.title('Valence')
    plt.savefig('val_chunks_flattened_res.jpg', dpi=1000)
    plt.show(block=True)

    plt.figure()
    plt.plot(aro_chunks_flattened_res_ds)
    plt.title('Arousal')
    plt.savefig('aro_chunks_flattened_res.jpg', dpi=1000)
    plt.show(block=True)
    plt.figure()
    plt.plot(aro_chunks_flattened_res_ds)
    plt.title('Arousal')
    plt.savefig('aro_chunks_flattened_res.jpg', dpi=1000)
    plt.show(block=True)
    # 

    #%%--------------------------------------

    # val_chunks_flattened.to_csv('val_chunks_flattened_res.csv', index=False)

    #%%----------------------------------------------------------------

    # Function to format y-axis labels with commas
    def format_with_commas(x, pos):
        return '{:,.0f}'.format(x)

    bin_width = 0.1
    bins = np.arange(1, 9.1, bin_width)  # The end value is 9.1 to include 9
    plt.figure()
    histogram_bins_val, bins, patches_val = plt.hist(val_chunks_flattened_res_ds, bins=bins, edgecolor='black', alpha=0.7)
    # Formatting the y-axis to include commas for thousands
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_with_commas))

    plt.xlabel('Labels', fontsize=14)
    plt.ylabel('Number of points per class', fontsize=12)
    plt.tight_layout() # Adjust layout to prevent cropping of labels
    plt.savefig('point_per_class_valence'+'_w'+str(window_seconds) +'_'+str(bin_width)+'.jpg', dpi=1000)
    plt.show(block=True)
    time.sleep(5)
    plt.close()

    bins = np.arange(1, 9.1, bin_width)  # The end value is 9.1 to include 9
    plt.figure()
    histogram_bins_aro, bins, patches_val = plt.hist(aro_chunks_flattened_res_ds, bins=bins, edgecolor='black', alpha=0.7)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_with_commas))
    plt.xlabel('Labels', fontsize=14)
    plt.ylabel('Number of points per class', fontsize=12)
    plt.tight_layout() # Adjust layout to prevent cropping of labels
    plt.savefig('point_per_class_arousal'+'_w'+str(window_seconds) +'_'+str(bin_width)+'.jpg', dpi=1000)
    plt.show(block=True)
    time.sleep(5)
    plt.close()

    #%%----------------------------------------------------------------
    # check number of data points

    print("aro_chunks_flattened_res (total nro points): ", aro_chunks_flattened_res_ds.shape[0])
    print("aro histogram_bins_val  (total nro points): ", histogram_bins_aro.sum())

    print("val_chunks_flattened_res (total nro points): ", val_chunks_flattened_res_ds.shape[0])
    print("val histogram_bins_val (total nro points): ", histogram_bins_val.sum())


    #%%----------------------------------------------------------------

    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time  # in seconds

    print("Elapsed time:", elapsed_time/(60), "minutes")
    print("Elapsed time:", elapsed_time/(60*60), "hours")

    
    

