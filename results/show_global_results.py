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

plt.xlabel('Window size (sec)')
plt.ylabel('Accuracy')
plt.xticks(window_size)  
plt.legend(loc='best')
plt.ylim(0.625, 0.8)  # Adjust the limits as needed
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


plt.xlabel('Window size (sec)')
plt.ylabel('Accuracy')
plt.xticks(window_size)  
plt.legend(loc='best')
plt.ylim(0.625, 0.8)  # Adjust the limits as needed
plt.grid()
plt.savefig('window_size_effect_valence'+'.jpg', dpi=600)
plt.show()


#%%----------------------------------------------------------------
# ZOOM around the optimal window size

# ----------------------------- Arousal  ----------------------------------------------------


overlap0 =   [0.71, 0.70,  0.69,  0.69,  0.70,  0.69,  0.69,  0.68,  0.67]  
overlap25 =  [0.72, 0.72,  0.72,  0.71,  0.73,  0.72,  0.72,  0.71,  0.70]
overlap50 =  [0.75, 0.759, 0.76,  0.756, 0.761, 0.743, 0.749, 0.74, 0.74]
window_size= [7,     8,     9,    10,    11,    12,    13,    14,   15]


plt.plot(window_size, overlap0, label='Overlap 0%')
plt.plot(window_size, overlap25,label='Overlap 25%')
plt.plot(window_size, overlap50, label='Overlap 50%')

plt.xlabel('Window size (sec)')
plt.ylabel('Accuracy')
plt.xticks(window_size)  # Set x-axis ticks at the center of each group
plt.legend(loc='best')
plt.ylim(0.66, 0.8)  # Adjust the limits as needed
plt.grid()
plt.savefig('window_size_effect_arousal_optimum'+'.jpg', dpi=600)
plt.show()

# ------------------------------------- Valence ----------------------------------------------------------------

# Data
overlap0 =   [0.70, 0.69, 0.70, 0.69, 0.70, 0.68, 0.70, 0.68, 0.67, 0.67, 0.67, 0.67, 0.68]
overlap25 =  [0.72, 0.72, 0.73, 0.72, 0.72, 0.72, 0.71, 0.72, 0.71, 0.71, 0.71, 0.70, 0.7]
overlap50 =  [0.74, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.76, 0.75, 0.74, 0.75, 0.75, 0.74]
window_size= [9,    10,    11,   12,   13,   14,   15,   16,   17,  18,   19,   20,   21 ]

plt.plot(window_size, overlap0, label='Overlap 0%')
plt.plot(window_size, overlap25,label='Overlap 25%')
plt.plot(window_size, overlap50, label='Overlap 50%')

plt.xlabel('Window size (sec)')
plt.ylabel('Accuracy')
plt.xticks(window_size)  # Set x-axis ticks at the center of each group
plt.legend(loc='best')
plt.ylim(0.66, 0.8)  # Adjust the limits as needed
plt.grid()
plt.savefig('window_size_effect_valence_optimum'+'.jpg', dpi=600)
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


# Data
overlap0 =   [1276, 418, 251, 176, 137, 111, 93,  80,  70,  63,  55,  52, 45, 000, 000]
overlap25 =  [1681, 555, 333, 237, 182, 149, 123, 107, 94,  80,  73,  68, 61, 000, 000]
overlap50 =  [2548, 832, 497, 350, 270, 219, 184, 158, 137, 122, 108, 99, 88, 000, 000]
window_size = [ 1,   3,   5,   7,   9,   11, 13,  15,  17,  19,  21,  23,  25,  27,  29]

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

plt.xlabel('Window size (sec)')
plt.ylabel('Number of samples')
# plt.title('Number of samples for different window sizes')
plt.xticks(x0 + bar_width, window_size)  # Set x-axis ticks at the center of each group
plt.legend(loc='best')
# plt.ylim(0.625, 0.8)  # Adjust the limits as needed

plt.savefig('number_instances'+'.jpg', dpi=600)
# Show plot
plt.show()


# zoom:

# Data
window_size =  window_size[2:]
x0 =  x0[2:]
x25 =  x25[2:]
x50 =  x50[2:]
overlap0 =  overlap0[2:]
overlap25 =  overlap25[2:]
overlap50 =  overlap50[2:]


plt.bar(x0, overlap0, width=bar_width, label='Overlap 0%')
plt.bar(x25, overlap25, width=bar_width, label='Overlap 25%')
plt.bar(x50, overlap50, width=bar_width, label='Overlap 50%')
plt.xlabel('Window size (sec)')
plt.ylabel('Number of samples')
plt.title('Number of samples for different window sizes (Zoom on greater window sizes)')
plt.xticks(x0 + bar_width, window_size)  # Set x-axis ticks at the center of each group
plt.legend(loc='best')
plt.ylim(0, 500)  # Adjust the limits as needed


plt.savefig('number_instances_zoom'+'.jpg', dpi=600)
# Show plot
plt.show()


#%%----------------------------------------------------------------


# Strong, weak and classic labeling schemes (thresholds are far from 5)
# scheme = 'bipartite', number_thresholds = 2, threshold = 5, L = 3, H = 7

# ML experiments planning - pc.xlsx (summary of results - Authors' access only)

# sources: 
# c_sldl_1_3_1
# c_sldl_1_3_2
# c_sldl_1_3_3
# c_sldl_1_3_4
# c_sldl_1_3_5
# c_sldl_1_3_6
# c_sldl_1_2_11
# c_sldl_1_2_12
# c_sldl_1_2_13



# ------------------------ Arousal --------------------

# Data
overlap0 =   [0.88, 0.8, 0.7 ]
overlap25 =  [0.89, 0.84, 0.73 ]
overlap50 =  [0.92, 0.86, 0.76]
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

plt.xlabel('Labeling Scheme')
plt.ylabel('Accuracy')
plt.xticks(x0 + bar_width, window_size)  # Set x-axis ticks at the center of each group
plt.legend(loc='best')
plt.ylim(0.65, 0.95)  # Adjust the limits as needed

plt.savefig('labeling_scheme_arousal'+'.jpg', dpi=600)
plt.show()

# ------------- Valence ----------------------------------------------------------------

# Data
overlap0 =   [0.90, 0.81, 0.68 ]
overlap25 =  [0.89, 0.81, 0.72]
overlap50 =  [0.92, 0.85, 0.76]
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

plt.xlabel('Labeling Scheme')
plt.ylabel('Accuracy')
plt.xticks(x0 + bar_width, window_size)  # Set x-axis ticks at the center of each group
plt.legend(loc='best')
plt.ylim(0.65, 0.95)  # Adjust the limits as needed

plt.savefig('labeling_scheme_valence'+'.jpg', dpi=600)
# Show plot
plt.show()



#%%----------------------------------------------------------------
# Try Linear DEAP / K-EmoCon features in CASE ds.
# To check if ACC increases with more instances and if linear features can extract relevant information from the signal (keeping the same features).

# source:
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

# Data 
# list Columns : window size: 3, 5, 8, 11, 16 sec, respectively. 

val_overlap0_acc_st =   [0.65, 0.65, 0.65, 0.65, 0.65]  # st _ statistical features
val_overlap0_uar_st =   [0.59, 0.60, 0.61, 0.61, 0.61]  
aro_overlap0_acc_st =   [0.64, 0.65, 0.65, 0.65, 0.64]  
aro_overlap0_uar_st =   [0.61, 0.62, 0.62, 0.62, 0.62]  

val_overlap0_acc_nl =   [0.71, 0.71, 0.70, 0.70, 0.68]  # nl _ non linear features
val_overlap0_uar_nl =   [0.64, 0.68, 0.67, 0.66, 0.65]  
aro_overlap0_acc_nl =   [0.71, 0.71, 0.70, 0.70, 0.68] 
aro_overlap0_uar_nl =   [0.66, 0.68, 0.68, 0.67, 0.65]  

val_overlap25_acc_st =   [0.67, 0.67, 0.66, 0.67, 0.67] 
val_overlap25_uar_st =   [0.62, 0.62, 0.62, 0.62, 0.63]
aro_overlap25_acc_st =   [0.68, 0.68, 0.66, 0.67, 0.67]  
aro_overlap25_uar_st =   [0.65, 0.64, 0.63, 0.64, 0.65]  

val_overlap25_acc_nl =   [0.73, 0.72, 0.73, 0.73, 0.72] 
val_overlap25_uar_nl =   [0.70, 0.69, 0.69, 0.69, 0.69]  
aro_overlap25_acc_nl =   [0.72, 0.73, 0.72, 0.73, 0.71] 
aro_overlap25_uar_nl =   [0.70, 0.7, 0.70, 0.70, 0.69]  

val_overlap50_acc_st =   [0.68, 0.68, 0.69, 0.69, 0.71] 
val_overlap50_uar_st =   [0.62, 0.63, 0.65, 0.65, 0.67]
aro_overlap50_acc_st =   [0.69, 0.68, 0.69, 0.71, 0.69]  
aro_overlap50_uar_st =   [0.64, 0.66, 0.67, 0.69, 0.66]  

val_overlap50_acc_nl =   [0.74, 0.74, 0.75, 0.75, 0.76]  # c_sldl_1_2, c_sldl_1_2_4, c_sldl_1_2_7 
val_overlap50_uar_nl =   [0.71, 0.71, 0.72, 0.72, 0.73]  
aro_overlap50_acc_nl =   [0.75, 0.75, 0.76, 0.76, 0.75] 
aro_overlap50_uar_nl =   [0.73, 0.73, 0.74, 0.74, 0.73]  

# 3 seconds window size
val_acc_st = [val_overlap0_acc_st[0], val_overlap25_acc_st[0] ,val_overlap50_acc_st[0]]
val_acc_nl = [val_overlap0_acc_nl[0], val_overlap25_acc_nl[0] ,val_overlap50_acc_nl[0]]

aro_acc_st = [aro_overlap0_acc_st[0], aro_overlap25_acc_st[0] ,aro_overlap50_acc_st[0]]
aro_acc_nl = [aro_overlap0_acc_nl[0], aro_overlap25_acc_nl[0] ,aro_overlap50_acc_nl[0]]


overlap = [0,0.25, 0.5]

#%%----------------------------------------------------------------

# 11 seconds window size
val_acc_st = [val_overlap0_acc_st[3], val_overlap25_acc_st[3] ,val_overlap50_acc_st[3]]
val_acc_nl = [val_overlap0_acc_nl[3], val_overlap25_acc_nl[3] ,val_overlap50_acc_nl[3]]

aro_acc_st = [aro_overlap0_acc_st[3], aro_overlap25_acc_st[3] ,aro_overlap50_acc_st[3]]
aro_acc_nl = [aro_overlap0_acc_nl[3], aro_overlap25_acc_nl[3] ,aro_overlap50_acc_nl[3]]


overlap = [0,0.25, 0.5]

# Define the width of each bar
bar_width = 0.2

# Create bar plots
plt.bar(x1, aro_acc_st, width=bar_width, label='Statistical feat.')
plt.bar(x2, aro_acc_nl, width=bar_width, label='Nonlinear')

plt.xlabel('Overlap')
plt.ylabel('Accuracy')
# plt.title('Arousal accuracy for window size = 11 sec  | Random Forest')
plt.xticks(x1 + bar_width, overlap)  # Set x-axis ticks at the center of each group
plt.legend(loc='best')
plt.ylim(0.6, 0.77)  # Adjust the limits as needed

plt.savefig('stats_vs_non_linear_11sw_arousal'+'.jpg', dpi=600)
# Show plot
plt.show()


#%%----------------------------------------------------------------
# 16 seconds window size
val_acc_st = [val_overlap0_acc_st[4], val_overlap25_acc_st[4] ,val_overlap50_acc_st[4]]
val_acc_nl = [val_overlap0_acc_nl[4], val_overlap25_acc_nl[4] ,val_overlap50_acc_nl[4]]

aro_acc_st = [aro_overlap0_acc_st[4], aro_overlap25_acc_st[4] ,aro_overlap50_acc_st[4]]
aro_acc_nl = [aro_overlap0_acc_nl[4], aro_overlap25_acc_nl[4] ,aro_overlap50_acc_nl[4]]


overlap = [0,0.25, 0.5]

# Define the width of each bar
bar_width = 0.2

# Create bar plots
plt.bar(x1, val_acc_st, width=bar_width, label='Statistical feat.')
plt.bar(x2, val_acc_nl, width=bar_width, label='Nonlinear')

plt.xlabel('Overlap')
plt.ylabel('Accuracy')
plt.xticks(x1 + bar_width, overlap)  # Set x-axis ticks at the center of each group
plt.legend(loc='best')
plt.ylim(0.6, 0.77)  # Adjust the limits as needed

plt.savefig('stats_vs_non_linear_16sw_valence'+'.jpg', dpi=600)
# Show plot
plt.show()




#%%----------------------------------------------------------------




