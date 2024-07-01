import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt,exp
import os
import pickle as pkl
from utils import fourier_threshold, wasserstein_compare, calculate_euuclidian_dist, kde
import scipy
import time
import math
import multiprocessing



data_vectors = np.load("./dataset_LensVector.npz")

dataset_array = data_vectors["data"]
results = []

def calc_euclid(i):
    a = dataset_array[i[0]]
    b = dataset_array[i[1]]
    return (np.sqrt(np.sum(np.square(a - b))))

param_array = []
for i in range(int(math.log2(len(dataset_array)))):
    for j in range(len(dataset_array)):
        param_array.append([i, j])


mypool = multiprocessing.Pool(32)


res_array = mypool.map(calc_euclid, param_array)
mypool.close()


print(np.real(np.median(res_array)))