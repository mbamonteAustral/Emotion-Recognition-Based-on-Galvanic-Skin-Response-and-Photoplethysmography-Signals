"""
main.py

main script. It calls processDEAPdata.py to apply different ALG to 
each participant data.
"""

#%% --------------------------------------------------
from processdataf import process_data
import numpy as np
from numpy import savetxt
from sys import platform
from openpyxl import load_workbook
import time
import tensorflow as tf

with tf.device('/CPU:0'):

    # Record the start time
    start_time = time.time()

    participants = np.arange(1, 31)  # participant from 1 to 30

    # participants = np.arange(1, 4)  # participant from 1 to 30

    # splitterseed = 12345
    # estimatorseed = 12345

    # index = 1 # usefull if first participant is not the number 1. Index ensure a first participant performance is loaded, independentedly participant number is 1 or any other.
    # if index = 1,  then a first participant is NOT loaded yet.

    #%% --------------------------------------------------
    # check if algorithm performance files existes:  
    import os.path

    alg_files_created = 0    # = 1 is algorithm performance files are already present  

    knn_file_exists = os.path.exists('knn_perm.csv')
    dt_file_exists = os.path.exists('dt_perm.csv')
    rf_file_exists = os.path.exists('rf_perm.csv')
    svm_file_exists = os.path.exists('svm_perm.csv')
    gbm_file_exists = os.path.exists('gbm_perm.csv')
    bddae_file_exists = os.path.exists('bddae_perm.csv')
    dummy_perm_exists = os.path.exists('dummy_perm.csv')

    if knn_file_exists and dt_file_exists and rf_file_exists and svm_file_exists and gbm_file_exists and bddae_file_exists and dummy_perm_exists:
        print("all algorithm performance files were created correctly before")
        alg_files_created = 1

        from numpy import loadtxt
        print("loading data already saved in algorithm performance files")
        # load array
        knn_perm = loadtxt('knn_perm.csv', delimiter=',')
        dt_perm = loadtxt('dt_perm.csv', delimiter=',')
        rf_perm = loadtxt('rf_perm.csv', delimiter=',')
        svm_perm = loadtxt('svm_perm.csv', delimiter=',')
        gbm_perm = loadtxt('gbm_perm.csv', delimiter=',')
        bddae_perm = loadtxt('bddae_perm.csv', delimiter=',')
        dummy_perm = loadtxt('dummy_perm.csv', delimiter=',')
        
            
    for i in participants:
        
        print("    ")
        print("    ")
        print("-------------------------------")
        print("participant: ", i)
        print("-------------------------------")
        print("    ")
        print("    ")
        try:
            performance, window_size, step, overlap, instances = process_data(str(i))
            # performance, window_size, step, overlap, instances = np.zeros((7, 10)), 3000, 1000,0.2, 100  # for testing main.py purspose only

            if (alg_files_created == 0):     #load first participant performance    
                knn_perm = performance[0]
                dt_perm = performance[1]
                rf_perm = performance[2]
                svm_perm = performance[3]
                gbm_perm = performance[4]
                bddae_perm = performance[5]
                dummy_perm = performance[6]

                # index = index + 1 
                alg_files_created = 1
                print("first participant performance loaded in numpy array")
            else:
                knn_perm = np.vstack([knn_perm,performance[0]])
                dt_perm = np.vstack([dt_perm,performance[1]])
                rf_perm = np.vstack([rf_perm,performance[2]])
                svm_perm = np.vstack([svm_perm,performance[3]])
                gbm_perm = np.vstack([gbm_perm,performance[4]])   
                bddae_perm = np.vstack([bddae_perm,performance[5]])   
                dummy_perm = np.vstack([dummy_perm,performance[6]])     
                print("last participant performance loaded in numpy array")
    
        except Exception as exc:
            print("-----------Exception computing performance ----------------")
            print(exc)
            print("-------------------------------------")

            
        print("    ")
        print("    ")
        print("-------------------------------")
        print("LAST participant's saved data: ")
        print(i)
        print("-------------------------------")
        print("    ")
        print("    ")
        

    # %% -----------------------------------
    # save performance in .csv file:

    print("Saving performance metrics in csv files...")
    savetxt('knn_perm.csv',knn_perm, delimiter=',')
    savetxt('dt_perm.csv',dt_perm, delimiter=',')
    savetxt('rf_perm.csv',rf_perm, delimiter=',')   
    savetxt('svm_perm.csv',svm_perm, delimiter=',')
    savetxt('gbm_perm.csv',gbm_perm, delimiter=',')
    savetxt('bddae_perm.csv',bddae_perm, delimiter=',')
    savetxt('dummy_perm.csv',dummy_perm, delimiter=',')

    # Introduce a delay of 5 seconds to give savetxt to really save each file and use it bellow.  
    time.sleep(5)

    print("Performance metrics already saved in csv files")

    #%% ----------------------------------------
    # Print results from simulations files (.csv)

    from functions import print_results_from_file 

    try:
        
        print_results_from_file(knn_perm, dt_perm, rf_perm, svm_perm, gbm_perm, bddae_perm, dummy_perm, print_results=True, show_plot=True)
    
    except Exception as exc:
        print("-----------Exception print_results_from_file ----------------")
        print(exc)
        print("-------------------------------------")

    import os
    print("-------------------------------------")
    print("Current folder:", os.getcwd())
    print("-------------------------------------")
    print("----- RESULTS ------")
    print("-------------------------------------")
    print("Window size (sec): ", window_size/1000)
    print("step (sec): ", step/1000)
    print("overlap: ", overlap)
    if overlap:
        perc_overlap = (window_size-step)/window_size
        print("perc. of overlap: ", (window_size-step)*100/window_size)
        print("overlap duration (sec): ", window_size*perc_overlap/1000)
    print("Number of windows / instances: ", instances)

    
    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time  # in seconds

    print("Elapsed time:", elapsed_time/(60), "minutes")
    print("Elapsed time:", elapsed_time/(60*60), "hours")

    #%% -------------------------
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------


    #%% ----------------------------------------
    # Print results from simulations files (.csv)

    # from functions import print_results_from_file 

    # print_results_from_file(print_results=True, show_plot=False)
    


