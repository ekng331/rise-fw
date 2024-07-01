import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt,exp
import os
import pickle as pkl
from utils import fourier_threshold
import multiprocessing

def createVectorsMulti(params):
    return createVectorsIndividual(params[0], params[1])

def createVectorsIndividual(path, size):
    dir = path
    img = cv2.imread(dir, 0)
    try:
        img = img.astype(np.float32)
    except:
        print(dir)
        img = img.astype(np.float32)
    img = cv2.resize(img, (size,size))
    #img_fft = np.fft.fft2(img)
        
    vector_horizontal = np.sum(img, axis=1)
    vector_vertical = np.sum(img, axis=0)

    return np.concatenate((vector_horizontal, vector_vertical))

def createListofParametersVectors(path, size, dir_list):
    name_array = np.array(dir_list)
    listofimgs = []
    for i in range(len(dir_list)):
        dir = dir_list[i]
        if "spr" in dir or dir[0] == '.':
            continue
        if dir[-3:] != "jpg":
            print(dir)
        res = path+"/"+dir
        listofimgs.append([res, size])
    return listofimgs


path = "/sharedata/fastdisk/nkng1/QSOlens/2021-GraL-ImageSet-gri/QSO"
dir_list = sorted([file for file in os.listdir(path)])

mypool = multiprocessing.Pool(32)
params = createListofParametersVectors(path, 256, dir_list)



res_array = np.array(mypool.map(createVectorsMulti, params))
print(res_array.shape)
mypool.close()

np.savez("dataset_Lens256Vector.npz", data=res_array)


mypool = multiprocessing.Pool(32)
params = createListofParametersVectors(path, 1000, dir_list)

res_array = np.array(mypool.map(createVectorsMulti, params))
print(res_array.shape)
mypool.close()

np.savez("dataset_Lens.npz", data=res_array)