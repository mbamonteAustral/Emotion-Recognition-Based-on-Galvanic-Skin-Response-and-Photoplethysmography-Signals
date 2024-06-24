#%%----------------------------------------------------------------

import os

current_folder = os.getcwd()
print("Current folder:", current_folder)


#%%----------------------------------------------------------------

import os

current_folder = os.path.abspath(__file__)
print("Current folder:", current_folder)
