#%%--------------------------------------------------------------------

import numpy as np

data = [
    [0.54543117, 0.73879028, 0.82335055, 0.69042624, 0.76113181, 0.40148478, 0.68481891, 0.73559954, 0.62833003, 0.68456477],
    [0.1849799, 0.57308856, 0.72596099, 0.38466169, 0.54674625, 0.03197599, 0.51198301, 0.69473609, 0.12257228, 0.43763944],
    [0.02464789, 0.50769231, 0.84977051, 0.05547002, 0.47357047, 0.44889493, 0.71288246, 0.74883821, 0.68698255, 0.71872594],
    [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [0.21725193, 0.6091926, 0.60697074, 0.59659237, 0.60187065, 0.08258416, 0.53331755, 0.66583477, 0.26128418, 0.46480206],
    [0.01432817, 0.50636004, 0.58053643, 0.1799216, 0.40468458, -0.00235925, 0.49918033, 0.73317556, 0, 0.42302054],
    [0.0345227, 0.51424123, 0.64182444, 0.15757235, 0.43245757, 0.25784237, 0.61806079, 0.67547332, 0.5373959, 0.60475278],
    [0.22660178, 0.59767391, 0.68144005, 0.45533326, 0.5651101, 0.00793141, 0.50315221, 0.65022949, 0.06267435, 0.40610744]
]

numpy_array = np.array(data)
print(numpy_array)


    #%% ----------------------------------------
    # Print results from simulations files (.csv)

    # from functions import print_results_from_file 

    # print_results_from_file(print_results=True, show_plot=False)

print("participants: ", np.shape(numpy_array))

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

