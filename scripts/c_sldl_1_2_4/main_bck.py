#%% ----------------------------------------
# # load numpy array from csv file

from numpy import loadtxt

# # load array
knn_perm = loadtxt('knn_perm.csv', delimiter=',')
print(knn_perm)

dt_perm = loadtxt('dt_perm.csv', delimiter=',')
print(dt_perm)

rf_perm = loadtxt('rf_perm.csv', delimiter=',')
print(rf_perm)

print("----- RESULTS ------")
print("-------------------------------------")
print("-rows: alg : KNN = 0; DA = 1; RF = 2; SVM = 3; GBM = 4; BDDAE = 5; DUMMY = 5")
print("columns:")
print("'val_cohen','val_uar', 'val_acc', 'val_gm',  'val_f1',  'aro_cohen','aro_uar', 'aro_acc', 'aro_gm',  'aro_f1'")
# print("-columns: [(val_cohen, val_uar, val_acc,val_gm, val_f1, aro_cohen, aro_uar,aro_acc,aro_gm, aro_f1)]")
print(np.round(performance, decimals=3))


# svm_perm = loadtxt('svm_perm.csv', delimiter=',')
# print(svm_perm)

# gbm_perm = loadtxt('gbm_perm.csv', delimiter=',')
# print(gbm_perm)


# #%% -------------------------
# # print algorithms performances:

# print("---------------------------")
# print("KNN performance:")
# print(knn_perm)
# print("KNN mean:")
# print(knn_perm.mean(axis=0))
# print("---------------------------")

# print("---------------------------")
# print("DT performance:")
# print(dt_perm)
# print("DT mean:")
# print(dt_perm.mean(axis=0))
# print("---------------------------")

# print("---------------------------")
# print("RF performance:")
# print(rf_perm)
# print("RF mean:")
# print(rf_perm.mean(axis=0))
# print("---------------------------")

# print("---------------------------")
# print("SVM performance:")
# print(svm_perm)
# print("SVM mean:")
# print(svm_perm.mean(axis=0))
# print("---------------------------")

# print("---------------------------")
# print("GBM performance:")
# print(gbm_perm)
# print("GBM mean:")
# print(gbm_perm.mean(axis=0))
# print("---------------------------")

# #%% -------------------------
# print("---------------------------")
# print("ALG Means: ")
# print("1) KNN, 2) DT, 3) RF, 4) SVM, 5) GBM, 6) DUMMY")
# print(knn_perm.mean(axis=0))
# print(dt_perm.mean(axis=0))
# print(rf_perm.mean(axis=0))
# print(svm_perm.mean(axis=0))
# print(gbm_perm.mean(axis=0))
# print("---------------------------")


# #%% ----------------------------------------
# # load numpy array from csv file

# from numpy import loadtxt
# # load array
# knndata = loadtxt('knn_perm.csv', delimiter=',')
# print(knndata)

# dtdata = loadtxt('dt_perm.csv', delimiter=',')
# print(dtdata)

# rfdata = loadtxt('rf_perm.csv', delimiter=',')
# print(rfdata)

# svmdata = loadtxt('svm_perm.csv', delimiter=',')
# print(svmdata)

# gbmdata = loadtxt('gbm_perm.csv', delimiter=',')
# print(gbmdata)

# #%%----------------------------------------------------------------
# # --- box plot - labels=('val_acc', 'aro_acc')
# # ----------------------------------------------------------------
# import matplotlib.pyplot as plt
# import numpy as np

# valaro=np.append(knndata[:,0:1],knndata[:,2:3],axis=1)
# plt.boxplot(valaro,labels=('val_acc', 'aro_acc'))
# plt.title('KNN performance')
# plt.show()

# plt.boxplot(knndata,labels=('val_acc', 'val_auc','aro_acc','aro_auc'))
# plt.title('KNN performance')
# plt.show()



# #%%----------------------------------------------------------------
# # --- box plot - labels=('val_acc', 'aro_acc')
# # ----------------------------------------------------------------

# valaro=np.append(dtdata[:,0:1],dtdata[:,2:3],axis=1)
# plt.boxplot(valaro,labels=('val_acc', 'aro_acc'))
# plt.title('DT performance')
# plt.show()

# plt.boxplot(dtdata,labels=('val_acc', 'val_auc','aro_acc','aro_auc'))
# plt.title('DT performance')
# plt.show()


# #%%----------------------------------------------------------------
# # --- box plot - labels=('val_acc', 'aro_acc')
# # ----------------------------------------------------------------

# valaro=np.append(rfdata[:,0:1],rfdata[:,2:3],axis=1)
# plt.boxplot(valaro,labels=('val_acc', 'aro_acc'))
# plt.title('RF performance')
# plt.show()

# plt.boxplot(rfdata,labels=('val_acc', 'val_auc','aro_acc','aro_auc'))
# plt.title('RF performance')
# plt.show()



# #%%----------------------------------------------------------------
# # --- box plot - labels=('val_acc', 'aro_acc')
# # ----------------------------------------------------------------

# valaro=np.append(svmdata[:,0:1],svmdata[:,2:3],axis=1)
# plt.boxplot(valaro,labels=('val_acc', 'aro_acc'))
# plt.title('SVM performance')
# plt.show()

# plt.boxplot(svmdata,labels=('val_acc', 'val_auc','aro_acc','aro_auc'))
# plt.title('SVM performance')
# plt.show()



# #%%----------------------------------------------------------------
# # --- box plot - labels=('val_acc', 'aro_acc')
# # ----------------------------------------------------------------

# valaro=np.append(gbmdata[:,0:1],gbmdata[:,2:3],axis=1)
# plt.boxplot(valaro,labels=('val_acc', 'aro_acc'))
# plt.title('GBM performance')
# plt.show()

# plt.boxplot(gbmdata,labels=('val_acc', 'val_auc','aro_acc','aro_auc'))
# plt.title('GBM performance')
# plt.show()


# # #%%----------------------------------------------------------------
# # # --- box plot - labels=('val_acc', 'val_auc','aro_acc','aro_auc')

# # import matplotlib.pyplot as plt

# # # plt.boxplot(knndata[:,1:5],labels=('val_acc', 'val_auc','aro_acc','aro_auc'))
# # plt.boxplot(knndata,labels=('val_acc', 'val_auc','aro_acc','aro_auc'))
# # plt.title('KNN performance')
# # plt.show()


# # #%%----------------------------------------------------------------
# # # --- box plot - labels=('val_acc', 'val_auc','aro_acc','aro_auc')

# # # plt.boxplot(dtdata[:,1:5],labels=('val_acc', 'val_auc','aro_acc','aro_auc'))
# # plt.boxplot(dtdata,labels=('val_acc', 'val_auc','aro_acc','aro_auc'))
# # plt.title('DT performance')
# # plt.show()

# # #%%----------------------------------------------------------------
# # # --- box plot - labels=('val_acc', 'val_auc','aro_acc','aro_auc')

# # plt.boxplot(rfdata,labels=('val_acc', 'val_auc','aro_acc','aro_auc'))
# # plt.title('RF performance')
# # plt.show()


# # #%%----------------------------------------------------------------
# # # --- box plot - labels=('val_acc', 'val_auc','aro_acc','aro_auc')

# # plt.boxplot(svmdata,labels=('val_acc', 'val_auc','aro_acc','aro_auc'))
# # plt.title('svm performance')
# # plt.show()


# # #%%----------------------------------------------------------------
# # # --- box plot - labels=('val_acc', 'val_auc','aro_acc','aro_auc')

# # plt.boxplot(gbmdata,labels=('val_acc', 'val_auc','aro_acc','aro_auc'))
# # plt.title('gbm performance')
# # plt.show()

# #%%----------------------------------------------------------------
# # --- ALG performance per participant

# participants = [1,2,4,5,6,7,8,9,10,11,12,15,19,16,20,21]  #16 participants

# for i in range(0,len(participants)):
#     print("participant n.", participants[i])
#     print("     val_acc -- val_auc -- aro_acc --- aro_auc")
#     print("KNN", knndata[i,:])
#     print("DT",dtdata[i,:])
#     print("RF", rfdata[i,:])  
#     print("SVM", svmdata[i,:])
#     print("GBM", gbmdata[i,:])
#     print("")



# participants = [1,2,4,5,6,7,8,9,10,11,12,15,19,16,20,21]  #16 participants

# for i in range(0,len(participants)):
#     print("participant n.", participants[i])
#     print("     val_acc -- val_auc -- aro_acc --- aro_auc")
#     print("KNN", knndata[i,:])
#     print("DT",dtdata[i,:])
#     print("RF", rfdata[i,:])  
#     print("SVM", svmdata[i,:])
#     print("GBM", gbmdata[i,:])
#     print("")

