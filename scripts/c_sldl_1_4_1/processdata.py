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
    participant = str(24)
        

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

# test windowing

# TEST PURPOSES:
window_size = int(3*1000)  # ms
perc_overlap = 0.25

# window_size = int(11*1000)  # ms
# perc_overlap = 0.5
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

combined_df_mapped = map_valence_arousal(combined_df_median, scheme = 'bipartite', number_thresholds = 1, threshold = 5, L = 3.6, H = 4.4)

X_windowed, y_median = build_dataset(combined_df_mapped)

# linear features (used in DEAP and K-EmoCon)
# xf = feature_extract_GSR_PPG_dataframe(X_windowed)

# xf = feature_extract_GSR_PPG_non_linear(X_windowed)

# JUST FOR TESTING PURPOSES:
xf_retrieved_pickle = pd.read_pickle('xf.pkl')
print("\nRetrieved DataFrame from pickle file:")
print(xf_retrieved_pickle)
xf = xf_retrieved_pickle.copy()

y_median = pd.read_pickle('y_median.pkl')
print(y_median)


#%%----------------------------------------------------------------
xf # feature vector (in DataFrame format)
y_median  # label vector (in DataFrame format)

gsr_ts_arousal, ppg_ts_arousal, gsr_ts_valence, ppg_ts_valence, valence, arousal = format_dataset_for_DL(X_windowed, y_median)

arousal_d_one_class, gsr_arousal_d_one_class, valence_d_one_class, gsr_valence_d_one_class = imbalance_test(valence, arousal, show_plots = False)

y_median = y_median.astype(int)

#%%----------------------------------------------------------------
# MLP with pytorch. 

# REPETITIONS = 10

REPETITIONS = 1

acc_array = np.zeros(REPETITIONS)
gmean_array = np.zeros(REPETITIONS)
f1_score_macro_array = np.zeros(REPETITIONS)
uar_array = np.zeros(REPETITIONS)
cohen_array = np.zeros(REPETITIONS)


# y = y_median.iloc[:,2]  # arousal
y = y_median.iloc[:,1]  # valence

# Splitting data into training and testing sets
from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(xf, y, test_size=0.2, random_state=42)

TEST_DS_SIZE = 0.2

from sklearn.model_selection import StratifiedShuffleSplit 
sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_DS_SIZE, random_state=None) # valence / arousal StratifiedShuffleSplit
sss.get_n_splits(xf, y)  #get train and test datasets. 

for train_index, test_index in sss.split(xf, y):  #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html
    print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = xf.iloc[train_index], xf.iloc[test_index]  
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]  #labels are common to GSR and PPG

print("(DL) TRAIN number of instances: ", y_train.shape[0])
print("(DL) TEST number of instances: ", y_test.shape[0])
print("(DL) Total number of instances (TRAIN+TEST): ", y_train.shape[0]+y_test.shape[0])



# gsr_ts = gsr_ts_arousal.copy()
# ppg_ts = ppg_ts_arousal.copy() 
# gsr_label_d = arousal.copy()

for i in range(0,REPETITIONS):
    print("Split Repetition number: ", i)


    # Convert DataFrame to PyTorch tensors
    X_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    batch_size = 32

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Create data loaders.
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break


    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # device = "cpu"


    print(X_train_tensor)

    # #%%----------------------------------------------------------------
    # Define your neural network model
    class NeuralNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(NeuralNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            # self.dropout = nn.Dropout(p=0.7)
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.sigmoid = torch.nn.Sigmoid()
            # self.softmax = torch.nn.Softmax()
            # self.threshold = torch.nn.Threshold(-0.1, -10)
        

        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            # out = self.dropout(out)
            out = self.fc2(out)
            out = self.sigmoid(out)
            # out = self.softmax(out)
            # out = self.threshold(out)
            return out

    # Define hyperparameters
    input_size = xf.shape[1]  # Number of features
    hidden_size = 64  # Number of neurons in the hidden layer
    output_size = 1  # Output size, assuming you're predicting one value

    # Instantiate the model
    model = NeuralNet(input_size, hidden_size, output_size)

    # Move the model to the appropriate device
    model.to(device)

    # Define loss function and optimizer
    loss_fn = nn.MSELoss()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) # lr=0.001
    # optimizer = optim.SGE(model.parameters(), lr=0.001) # lr=0.001

    #%%----------------------------------------------------------------
    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):

            X, y = X.to(device), y.to(device)

            if torch.cuda.is_available(): 
                print(f"Using {device} device")

            # Compute prediction error
            pred = model(X)

            y = y.view_as(pred)
        
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # print("batch (count): ", batch)

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                # print("batch modulus: ", batch+1)
                # print("len(X) modulus: ", len(X))

                # print("current (train method): ", current)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    #%%----------------------------------------------------------------
    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        prediction_array = np.array([])
        y_array = np.array([])
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)

                # print("X en test: ", X)

                pred = model(X)
                # Reshape the target tensor to match the shape of the predicted output tensor
                y = y.view_as(pred)
                test_loss += loss_fn(pred, y).item()
                # print("test_loss: ", test_loss)

                print("pred: ", pred)
                # print("pred gt5: ", (pred >= 0.2).float())
                # correct = ((pred >= 0.5).float() == y).type(torch.float).sum().item()
                # print("correct: ", correct)
                # print("((pred >= 0.5).float() == y): ", ((pred >= 0.5).float() == y).float())
                # print((pred >= 0.5).float())
                correct += ((pred >= 0.5).float() == y).type(torch.float).sum().item()
                prediction = (pred >= 0.5).float().numpy()
                # prediction = (pred >= 0.5).float().tolist()
                # prediction = prediction.flatten()


                # prediction_array.append((pred >= 0.5).float().tolist())
                # prediction_array.append(prediction.tolist())
                # prediction_array.append(prediction)

                
                y_array = np.vstack((y_array, y)) if y_array.size else y
                prediction_array = np.vstack((prediction_array, prediction)) if prediction_array.size else prediction

                # prediction_array = np.reshape(prediction_array,(np.shape(prediction_array)[1],))
                
                print("np.shape(prediction_array)", np.shape(prediction_array))
                # correct += ((pred >= 0.5).float() == y).type(torch.float).sum().item()
                # correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        # print("correct: ", correct)
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        return prediction_array, y_array


    # #%%----------------------------------------------------------------
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        prediction_array, y_array = test(test_dataloader, model, loss_fn)
        
        print("")
        # print(np.shape(np.array(prediction)))
    print("Done!")

    [cohen_array[i], uar_array[i], acc_array[i], gmean_array[i], f1_score_macro_array[i]] = calc_metrics2(y_array, prediction_array)

    # if arousal_flag:
    #     [cohen_array[i], uar_array[i], acc_array[i], gmean_array[i], f1_score_macro_array[i]] = arousal_bimodal_deep_denoising_AE(convolutionalGSR, convolutionalPPG, x_trainGSR, x_testGSR, x_trainPPG, x_testPPG, y_train, y_test) 
    # else: 
    #     [cohen_array[i], uar_array[i], acc_array[i], gmean_array[i], f1_score_macro_array[i]] = valence_bimodal_deep_denoising_AE(convolutionalGSR, convolutionalPPG, x_trainGSR, x_testGSR, x_trainPPG, x_testPPG, y_train, y_test) 
    # print("-------- Model Performance ----------: ")    
    # print("accuracy: ", acc_array)
    # print("gmean: ", gmean_array)
    # print("f1_score: ", f1_score_macro_array)    
    # print("UAR: ", uar_array)
    # print("Cohen Kappa score: ", cohen_array)

    # np.array([np.nanmean(cohen_array), np.nanmean(uar_array), np.nanmean(acc_array),np.nanmean(gmean_array),np.nanmean(f1_score_macro_array)],dtype='float')


#%%----------------------------------------------------------------

# # ---- ALG TEST -----
# knn = KNeighborsClassifier(n_neighbors=5)
# dt = DecisionTreeClassifier()
# rf = RandomForestClassifier(n_estimators=5)
# svm = SVC(C=0.1)
# gbm = GradientBoostingClassifier(n_estimators=5)
# performance = alg_performance_eval(xf, y_median, knn, dt, rf, svm, gbm, arousal_d_one_class, valence_d_one_class, gsr_arousal_d_one_class, gsr_valence_d_one_class, gsr_ts_arousal, ppg_ts_arousal, arousal, gsr_ts_valence, ppg_ts_valence, valence)

# print("-------------------------------------")
# print("-------------------------------------")
# print("participant: ", participant)
# print("-------------------------------------")
# print("----- RESULTS ------")
# print("-------------------------------------")
# print("Window size (sec): ", window_size/1000)
# print("step (sec): ", step/1000)
# print("overlap: ", overlap)
# if overlap:
#     perc_overlap = (window_size-step)/window_size
#     print("perc. of overlap: ", (window_size-step)*100/window_size)
#     print("overlap duration (sec): ", window_size*perc_overlap/1000)
# print("Number of windows / instances: ", X_windowed.shape[0])
# print("-rows: alg : KNN = 0; DT = 1; RF = 2; SVM = 3; GBM = 4; BDDAE = 5; DUMMY = 5")
# print("columns:")
# print("'val_cohen','val_uar', 'val_acc', 'val_gm',  'val_f1',  'aro_cohen','aro_uar', 'aro_acc', 'aro_gm',  'aro_f1'")
# print("-------------------------------------------------------------")
# print("  v_c   v_u   v_a   v_g   v_f1  a_c   a_u   a_a   a_g   a_f1")
# # print("-columns: [(val_cohen, val_uar, val_acc,val_gm, val_f1, aro_cohen, aro_uar,aro_acc,aro_gm, aro_f1)]")
# print(np.round(performance, decimals=3))

