import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt,exp
import os
import pickle as pkl
from utils import fourier_threshold
import multiprocessing
import gzip


def createFouriersMulti(params):
    return createFouriersIndividual(params[0], params[1], params[2], params[3], params[4])

def createFouriersIndividual(path, threshold, size, section, max_size):
    dir = path
    img = cv2.imread(dir, 0)
    img = img.astype(np.float32)
    img = cv2.resize(img, (size,size))
    img_fft = np.fft.fft2(img)
    img_thresholded = fourier_threshold(np.real(img_fft), threshold)
    img_thresholded = np.sort(np.abs((img_thresholded).flatten()))
    return img_thresholded


def createListofParameters(path, threshold, size, section, max_size, dir_list):
    name_array = np.array(dir_list)
    listofimgs = []
    for i in range(len(dir_list)):
        dir = dir_list[i]
        if "spr" in dir or dir[0] == '.':
            continue
        res = path+"/"+dir
        listofimgs.append([res, threshold, size, section, max_size])
    return listofimgs

def createFouriers(path, threshold, size, section, max_size, dir_list):
    coeffs_array = []
    name_array = np.array(dir_list)
    for i in range(len(dir_list)):
        if i >= (section-1)*max_size and i < section*max_size:
            dir = dir_list[i]
            if "spr" in dir or dir[0] == '.':
                continue
            img = cv2.imread(path+"/"+dir, 0)
            img = img.astype(np.float32)
            img = cv2.resize(img, (size,size))
            img_fft = np.fft.fft2(img)
            img_thresholded = fourier_threshold(np.real(img_fft), threshold)
            img_thresholded = np.sort(np.abs((img_thresholded).flatten()))
            coeffs_array.append(img_thresholded)

    pad_length = 0
    for i in coeffs_array:
        if i.shape[0] > pad_length:
            pad_length = i.shape[0]
    for i in range(len(coeffs_array)):
        coeffs_array[i] = np.pad(coeffs_array[i], (0, pad_length - coeffs_array[i].shape[0]), constant_values=-1)
    coeffs_array = np.array(coeffs_array)
        
    
    return coeffs_array, name_array



path = "/sharedata/fastdisk/nkng1/QSOlens/2021-GraL-ImageSet-gri/QSO"

dir_list = sorted(os.listdir(path))

if False:
    dataset_array, name_array = createFouriers(path, .95, 256, 1, 10000, dir_list)
    print(name_array[0])
    np.savez_compressed("/sharedata/fastdisk/nkng1/dataset_Lens256.npz", data=dataset_array, names=name_array)

name_array = np.array(dir_list)


#for i in range(0, 4):
print("here")
mypool = multiprocessing.Pool(32)
params = createListofParameters(path, .95, 1000, 1, 10000, dir_list)




res_array = mypool.map(createFouriersMulti, params)
mypool.close()
print("here2")
coeffs_array = res_array
'''
pad_length = 0
for i in coeffs_array:
    if i.shape[0] > pad_length:
        pad_length = i.shape[0]

print("here 2.5")     
for i in range(len(coeffs_array)):
    if i % 500 == 0:
        print(i)
    coeffs_array[i] = np.pad(coeffs_array[i], (0, pad_length - coeffs_array[i].shape[0]), constant_values=-1)
    '''
print("here2.5 ", len(coeffs_array), len(coeffs_array[0]))
#coeffs_array = np.array(coeffs_array)
#print("shape ", np.shape(coeffs_array))
print("here3")


#dataset_array, name_array = createFouriers(path, .95, 1000, 1, 10000)

print(name_array[0])

#np.savez_compressed("/sharedata/fastdisk/nkng1/dataset_Lens.npz", data=coeffs_array, names=name_array)

f = gzip.GzipFile("/sharedata/fastdisk/nkng1/dataset_Lens.npz.gz", "w")
np.save(file=f, arr=coeffs_array)
f.close()