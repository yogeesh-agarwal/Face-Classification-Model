import os
import cv2
import pickle
import numpy as np
import pandas as pd

def load_pickle(filepath):
    with open(filepath , "rb") as f:
        return pickle.load(f)

def split_data(image_names):
    train_data = []
    val_data = []
    test_data = []

    for index in range(len(image_names)):
        if index >= 0 and index <= 162770:
            train_data.append(image_names[index])
        elif index >= 162771 and index <= 182637:
            val_data.append(image_names[index])
        else:
            test_data.append(image_names[index])

    return train_data, val_data , test_data

if __name__ == "__main__":
    image_names = load_pickle("./data/image_names.pickle")
    train_data , val_data , test_data = split_data(image_names)

    with open("./data/celeba_train_file.pickle" , "wb") as content:
        pickle.dump(train_data , content , protocol =  pickle.HIGHEST_PROTOCOL)
    with open("./data/celeba_val_file.pickle" , "wb") as content:
        pickle.dump(val_data , content , protocol =  pickle.HIGHEST_PROTOCOL)
    with open("./data/celeba_test_file.pickle" , "wb") as content:
        pickle.dump(test_data , content , protocol =  pickle.HIGHEST_PROTOCOL)

    print("training , validation and testing data are stored in pickle files")
