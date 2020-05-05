import numpy as np
import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

def copy_files(src, dest):
    src_files = os.listdir(src)
    for file_name in src_files:
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dest)

def find_sepsis_file(data_path_dir):
    id_nosepsis = []
    id_sepsis = []
    for psv in os.listdir(data_path_dir):
        pid = pd.read_csv(os.path.join(data_path_dir, psv), sep='|')
        if 1 in np.array(pid.SepsisLabel):
            id_sepsis.append(psv)
        else:
            id_nosepsis.append(psv)
    return (id_nosepsis, id_sepsis)

if __name__ == "__main__":
    data_path_A = "./data/training_setA/"
    data_path_B = "./data/training_setB/"
    data_path = "./data/all_dataset/"
    copy_files(data_path_A, data_path)
    copy_files(data_path_B, data_path)

    # divide a total of 40,336 populations into septic/no-septic (2,932/37,404) patients
    id_nosepsis, id_sepsis = find_sepsis_file(data_path)
    # development dateset (34,285 patients, 2,492 septic & 31,793 non-septic)
    # validation dataset (6,051 patients, 440 septic & 5,611 non-septic)
    train_nosepsis, test_nosepsis = train_test_split(id_nosepsis, test_size=0.15, random_state=12306)
    train_sepsis, test_sepsis = train_test_split(id_sepsis, test_size=0.15, random_state=12306)
    test_set = np.append(test_nosepsis, test_sepsis)

    np.save("./data/train_nosepsis.npy", train_nosepsis)
    np.save("./data/train_sepsis.npy", train_sepsis)
    np.save("./data/test_set.npy", test_set)
