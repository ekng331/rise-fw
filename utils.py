import numpy as np
import scipy
from sklearn.neighbors import KernelDensity
import statistics
import math

def fourier_threshold(img_orig, cutoff):
    cutoff2 = np.quantile(img_orig, cutoff)
    img_ret = np.where(np.abs(img_orig) > cutoff2, img_orig, 0)
    #counts, b = np.histogram(np.arcsinh(np.abs(img_ret.flatten())), bins=50, density=True)
    return img_ret

def wasserstein_compare(img_thresholded, dataset_array, no):
    wasserstein_distances = []
    i = 0
    for d in dataset_array:
        print(i, d[d>0].shape, img_thresholded[img_thresholded > 0].shape)
        i+=1
        wasserstein_distances.append(scipy.stats.wasserstein_distance(img_thresholded[img_thresholded > 0], d[d > 0]))
    
    wasserstein_distances = np.array(wasserstein_distances)
    return np.sort(wasserstein_distances)[:no], np.argsort(wasserstein_distances)[:no]
    '''
    print("number here", wasserstein_distances)
    ten_least = np.argsort(wasserstein_distances)[:no]
    return ten_least
    '''

def calculate_euuclidian_dist(img_thresholded, dataset_array):
    #print(dataset_array.shape, img_thresholded.shape)
    distances = np.sqrt(np.sum(np.square(img_thresholded - dataset_array), axis=1))
    #print(distances.shape)
    twenty_least = np.argsort(distances)[0:20]
    b = np.sort(distances)[0:20]
    print(twenty_least, b)
    return twenty_least

#do this kde beforehand in database
def kde(dataset_array):
    kde_array = []
    
    kde = KernelDensity(bandwidth='silverman', kernel='gaussian')

    for i in range(0, dataset_array.shape[0]):
        kde = kde.fit(dataset_array[i][:, None])
        kde_array.append(kde.sample(10000))
    kde_array = np.array(kde_array)
    return kde_array