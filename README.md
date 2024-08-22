# Determining the Optimal Window Duration to Enhance Emotion Recognition Based on Galvanic Skin Response and Photoplethysmography Signals

This repository contains the simulations necessary to reproduce our article using the CASE dataset.

https://www.mdpi.com/2079-9292/13/16/3333

Article authors:
Marcos F. Bamonte,
Marcelo Risk,
Victor Herrero.

## Citation
Bamonte, M.F.; Risk, M.; Herrero, V. Determining the Optimal Window Duration to Enhance Emotion Recognition Based on Galvanic Skin Response and Photoplethysmography Signals. Electronics 2024, 13, 3333. https://doi.org/10.3390/electronics13163333

##  CASE Dataset
CASE Dataset is made publicly available at https://gitlab.com/karan-shr/case_dataset.

## Scripts (folder)

Each folder contains one specific experiment or simulation. 

Each folder contains:
* python scripts
* csv files contain the metrics yielded by each algorithm for each participant in the CASE dataset. 

Python scripts:
* functions.py (Methods)
* definitions.py (Simulation definitions)
* processdataf.py (Trains machine learning models and evaluates performance metrics for a particular participant)
* **main.py** (Main running file. Trains and evaluates performance metrics for all participants. Generates .csv files with results)
* show_csv_results.py (Evaluates performance metrics for all participants. Requires .csv files generated by main.py)

To run the experiment, execute the main.py file. 

Please see the required packages to run the simulations properly ([requirements.md](https://github.com/mbamonteAustral/Emotion-Recognition-Based-on-Galvanic-Skin-Response-and-Photoplethysmography-Signals/blob/000b5ca78c06d6f1770bd48680095dccb16708b4/requirements.md)).

To run the scripts, the CASE dataset should be downloaded first. Suppose the downloaded folder is case_dataset. Then, three subfolders should be located within the downloaded dataset folder:

* physiological  (relative path: case_dataset\data\interpolated\physiological)
* annotations  (relative path: case_dataset\data\interpolated\annotations)
* metadata (relative path: case_dataset\metadata)

These paths should then be replaced conveniently in the **processdataf.py** code::

    if platform == "linux" or platform == "linux2":

        physiological_base = r"/home/marcos/datasets/case/physiological/sub_"  # replace with current downloaded physiological folder
        annotation_base = r"/home/marcos/datasets/case/annotations/sub_" # replace with current downloaded annotation folder
        seqs_order_num_base = r"//home/marcos/datasets/case/seqs_"    # replace with current downloaded root folder
    elif platform == "win32":
        if socket.gethostname() == "LAPTOP-R7AHG17P":  # pc gamer (Javier)
            print("corriendo en PC gamer:")
            physiological_base = r"C:\Users\Javier\Desktop\CASE_full\CASE_full\data\interpolated\physiological\sub_" # replace with current physiological folder
            annotation_base = r"C:\Users\Javier\Desktop\CASE_full\CASE_full\data\\interpolated\annotations\sub_"  # replace with current downloaded annotation folder
            seqs_order_num_base = r"C:\Users\Javier\Desktop\CASE_full\CASE_full\metadata\seqs_"  # replace with current downloaded root folder
            # Set the default encoding
            sys.stdout.reconfigure(encoding='utf-8')
        else:
            physiological_base = r"C:\Users\mbamo\Desktop\Datasets\CASE\interpolated\physiological\sub_"  # replace with current physiological folder
            annotation_base = r"C:\Users\mbamo\Desktop\Datasets\CASE\interpolated\annotations\sub_"  # replace with current downloaded annotation folder
            seqs_order_num_base = r"C:\Users\mbamo\Desktop\Datasets\CASE\metadata\seqs_"   # replace with current downloaded root folder



## Results (folder)

The folder contains figures summarizing the results.
See the README file in the Results folder

