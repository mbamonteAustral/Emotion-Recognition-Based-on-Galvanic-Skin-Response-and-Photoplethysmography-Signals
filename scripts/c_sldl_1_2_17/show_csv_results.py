"""
show_simulated_results.py
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


    #%% ----------------------------------------
    # Print results from simulations files (.csv)

    # from functions import print_results_from_file 

    def print_results_from_file(print_results=True, show_plot=True):
        """
        Uploads data from .csv files containing previous simulation results, 
        calculates mean metrics and standard deviations, 
        finally displays their corresponding box plots.
        """

        def plot_box_plot(data, alg_name='KNN'):

            plt.figure()
            valaro=np.append(data[:,2:3],data[:,7:8],axis=1)
            plt.boxplot(valaro,labels=('val_acc', 'aro_acc'))
            plt.title(alg_name+' performance (V/A)')
            plt.savefig(alg_name+'_VA'+'.png')
            # plt.show()

            plt.figure()
            plt.boxplot(data,labels=('v_c', 'v_u','v_a', 'v_g', 'v_f1', 'a_c', 'a_u','a_a', 'a_g', 'a_f1',))
            plt.title(alg_name+' performances')
            plt.savefig(alg_name+'_perfomances'+'.png')
            # plt.show()


        from numpy import loadtxt
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        current_folder = os.getcwd()

        # load data
        knn_perm = loadtxt(current_folder+'\\'+'knn_perm.csv', delimiter=',')
        dt_perm = loadtxt(current_folder+'\\'+'dt_perm.csv', delimiter=',')
        rf_perm = loadtxt(current_folder+'\\'+'rf_perm.csv', delimiter=',')
        svm_perm = loadtxt(current_folder+'\\'+'svm_perm.csv', delimiter=',')
        gbm_perm = loadtxt(current_folder+'\\'+'gbm_perm.csv', delimiter=',')
        bddae_perm = loadtxt(current_folder+'\\'+'bddae_perm.csv', delimiter=',')
        dummy_perm = loadtxt(current_folder+'\\'+'dummy_perm.csv', delimiter=',')

        if print_results:

            # #%% -------------------------
            # print algorithms performances from .csv files:

            print("---------------------------")
            print("KNN performance:")
            print(knn_perm)
            print("KNN mean:")
            print(np.nanmean(knn_perm,axis=0))

            print("---------------------------")

            print("---------------------------")
            print("DT performance:")
            print(dt_perm)
            print("DT mean:")
            print(np.nanmean(dt_perm,axis=0))

            print("---------------------------")

            print("---------------------------")
            print("RF performance:")
            print(rf_perm)
            print("RF mean:")
            print(np.nanmean(rf_perm,axis=0))

            print("---------------------------")

            print("---------------------------")
            print("SVM performance:")
            print(svm_perm)
            print("SVM mean:")
            print(np.nanmean(svm_perm,axis=0))

            print("---------------------------")

            print("---------------------------")
            print("GBM performance:")
            print(gbm_perm)
            print("GBM mean:")
            print(np.nanmean(gbm_perm,axis=0))

            print("---------------------------")

            print("---------------------------")
            print("BDDAE performance:")
            print(bddae_perm)
            print("BDDAE mean:")
            print(np.nanmean(bddae_perm,axis=0))

            print("---------------------------")

            print("---------------------------")
            print("DUMMY performance:")
            print(dummy_perm)
            print("DUMMY mean:")
            print(np.nanmean(dummy_perm,axis=0))

            print("---------------------------")

            #%%----------------------------------------------------------------
            # build final performace vector
            performance_final = np.array([(0.0, 0.0,0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,0.0)]) # store ALG metrics
            performance_final[-1] = np.nanmean(knn_perm,axis=0)
            performance_final = np.vstack([performance_final,np.nanmean(dt_perm,axis=0)])
            performance_final = np.vstack([performance_final,np.nanmean(rf_perm,axis=0)])
            performance_final = np.vstack([performance_final,np.nanmean(svm_perm,axis=0)])
            performance_final = np.vstack([performance_final,np.nanmean(gbm_perm,axis=0)])
            performance_final = np.vstack([performance_final,np.nanmean(bddae_perm,axis=0)])
            performance_final = np.vstack([performance_final,np.nanmean(dummy_perm,axis=0)])

            import os
            print("Current folder:", os.getcwd())
            print("-------------------------------------")
            print("----- RESULTS from .csv files ------")
            print("-------------------------------------")
            print("---------------------------")
            print("ALG Means: ")
            print("-rows: alg : KNN = 0; DT = 1; RF = 2; SVM = 3; GBM = 4; BDDAE = 5; DUMMY = 5")
            print("columns:")
            print("'val_cohen','val_uar', 'val_acc', 'val_gm',  'val_f1',  'aro_cohen','aro_uar', 'aro_acc', 'aro_gm',  'aro_f1'")
            print("-------------------------------------------------------------")
            print("  v_c   v_u   v_a   v_g   v_f1  a_c   a_u   a_a   a_g   a_f1")
            print(np.round(performance_final, decimals=3))

            sd_knn = np.round(np.nanstd(knn_perm, axis=0), decimals=3)
            sd_dt = np.round(np.nanstd(dt_perm, axis=0), decimals=3)
            sd_rf = np.round(np.nanstd(rf_perm, axis=0), decimals=3)
            sd_svm = np.round(np.nanstd(svm_perm, axis=0), decimals=3)
            sd_gbm = np.round(np.nanstd(gbm_perm, axis=0), decimals=3)
            sd_bddae = np.round(np.nanstd(bddae_perm, axis=0), decimals=3)
            sd_dummy = np.round(np.nanstd(dummy_perm, axis=0), decimals=3)

            print("-------------------------------------------------------------")
            print("Standard Deviation of each metric:")
            std_dev = np.array([(0.0, 0.0,0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,0.0)]) # store ALG metrics
            std_dev [-1] = sd_knn
            std_dev = np.vstack([std_dev,sd_dt])
            std_dev = np.vstack([std_dev,sd_rf])
            std_dev = np.vstack([std_dev,sd_svm])
            std_dev = np.vstack([std_dev,sd_gbm])
            std_dev = np.vstack([std_dev,sd_bddae])
            std_dev = np.vstack([std_dev,sd_dummy])
            print(std_dev)


            # Print the result after handling division by zero
            np.set_printoptions(linewidth=np.inf)
            with np.errstate(divide='ignore', invalid='ignore'):
                result = np.round(np.where(performance_final == 0, 0, std_dev * 100 / performance_final),decimals=0)
                print("-------------------------------------------------------------")
                print("std_dev/mean [percentage]:")
                print(result)        

        if show_plot:
            # --- box plot ------
            plot_box_plot(knn_perm, alg_name='KNN')
            plot_box_plot(dt_perm, alg_name='DT')
            plot_box_plot(rf_perm, alg_name='RF')
            plot_box_plot(svm_perm, alg_name='SVM')
            plot_box_plot(gbm_perm, alg_name='GBM')
            plot_box_plot(bddae_perm, alg_name='BDDAE')
            plot_box_plot(dummy_perm, alg_name='DUMMY')



    print_results_from_file(print_results=True, show_plot=True)
    


