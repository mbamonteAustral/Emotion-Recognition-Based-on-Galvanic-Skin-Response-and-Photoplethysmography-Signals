#%%--------------------------------------------------------------------

import numpy as np

data = [
    [0.52420996, 0.7322619, 0.81341463, 0.67771893, 0.75195277, 0.386456, 0.67947059, 0.7253194, 0.6225389, 0.67627956],
    [0.04180589, 0.51605722, 0.68890825, 0.13253979, 0.44567456, 0.0468863, 0.51750663, 0.7033101, 0.13867505, 0.44740064],
    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [-0.01005213, 0.49980159, 0.65069686, 0.13067735, 0.42943796, 0.06208993, 0.526125, 0.62439024, 0.2909157, 0.46275945],
    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [0.03843844, 0.5152381, 0.8445993, 0.05690426, 0.47988516, 0.46682553, 0.72301716, 0.75603949, 0.69918229, 0.7281602],
    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [0.24232762, 0.62187229, 0.6195122, 0.60771109, 0.61365492, 0.12420728, 0.55136396, 0.66759582, 0.33557039, 0.5032789],
    [0.03118802, 0.51361275, 0.58135889, 0.25752137, 0.43964112, -0.00468085, 0.49833333, 0.73199768, 0, 0.42260903],
    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [0.25661625, 0.60645604, 0.73455285, 0.46958759, 0.59589777, 0.46900402, 0.73928571, 0.73931475, 0.72483241, 0.73353522],
    [0.02772856, 0.51148148, 0.64117305, 0.15950938, 0.43309811, 0.29006785, 0.63527828, 0.68896632, 0.5667688, 0.62704869],
    [0.1938689, 0.58759446, 0.67003484, 0.42542563, 0.54931703, 0.02855879, 0.51148148, 0.64831591, 0.08815462, 0.41635647]
]

numpy_array = np.array(data)
print(numpy_array)


    #%% ----------------------------------------
    # Print results from simulations files (.csv)

    # from functions import print_results_from_file 

    # print_results_from_file(print_results=True, show_plot=False)

print("instances: ", np.shape(numpy_array))

#%%----------------------------------------------------------------
# Calculate the standard deviation of each column
np.set_printoptions(linewidth=np.inf)
mean = np.round(np.nanmean(numpy_array, axis=0), decimals=3)
print("Mean of each column:")
print("  v_c   v_u   v_a   v_g   v_f1  a_c   a_u   a_a   a_g   a_f1")
print(mean)


std_dev = np.round(np.nanstd(numpy_array, axis=0), decimals=3)
print("  v_c   v_u   v_a   v_g   v_f1  a_c   a_u   a_a   a_g   a_f1")
print("Standard Deviation of each column:")
print(std_dev)


# Print the result after handling division by zero
with np.errstate(divide='ignore', invalid='ignore'):
    result = np.round(np.where(mean == 0, 0, std_dev * 100 / mean), decimals=2)
    print("std_dev*100/mean:")
    print("  v_c   v_u   v_a   v_g   v_f1  a_c   a_u   a_a   a_g   a_f1")
    print(result)

