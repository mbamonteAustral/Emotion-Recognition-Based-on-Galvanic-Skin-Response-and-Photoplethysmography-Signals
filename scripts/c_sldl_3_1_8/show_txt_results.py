#%%--------------------------------------------------------------------

import numpy as np

data = [
    [0.48010969, 0.70914731, 0.79665584, 0.65253124, 0.72560844, 0.34496554, 0.65697318, 0.71357143, 0.58323118, 0.64973653],
    [0.14315435, 0.55861713, 0.71704545, 0.34107034, 0.52347954, 0.0134401, 0.50507264, 0.6863961, 0.10509772, 0.43207305],
    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [0.01734317, 0.50555556, 0.84688312, 0.03333333, 0.46846098, 0.3919813, 0.68353484, 0.72824675, 0.63453352, 0.68233167],
    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [0.18758552, 0.59217226, 0.59087662, 0.58913104, 0.58996553, 0.10428338, 0.5399457, 0.66126623, 0.29990208, 0.48393296],
    [0.00540316, 0.50261446, 0.57655844, 0.16586971, 0.40532001, 0, 0.5, 0.73516234, 0, 0.42367573],
    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [0.0383129, 0.5159127, 0.6413961, 0.14894209, 0.43009605, 0.29199384, 0.63318221, 0.6937013, 0.5445759, 0.61805687],
    [0.28580334, 0.62837408, 0.69724026, 0.53866454, 0.61415002, 0.16431682, 0.57089379, 0.69181818, 0.36512804, 0.531544]
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

