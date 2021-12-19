import os
import cv2
import glob
import argparse
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans

def get_list_of_files(dir, ext):
    return glob.glob(os.path.join(dir, ext))


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='Images dir')
parser.add_argument('--output', type=str, help='Numpy dir')
parser.add_argument('--hog', action = 'store_true', help = 'If True, apply HoG')
parser.add_argument('--kmeans', type=int, default=2, help='k clusters for k-means')
parser.add_argument('--clusters', type=str, help='Clusters dir')
opt = parser.parse_args()

Path(opt.output).mkdir(parents=True, exist_ok=True)
Path(opt.clusters).mkdir(parents=True, exist_ok=True)
for k in range(opt.kmeans):
    Path(os.path.join(opt.clusters, str(k))).mkdir(parents=True, exist_ok=True)

list_of_images = get_list_of_files(opt.data, '*.png')
if opt.hog:
    hog = cv2.HOGDescriptor()
    for i in list_of_images:
        image = cv2.imread(i, 0)
        edges = cv2.Canny(image, 100, 200)
        hog_features = hog.compute(edges)
        hog_features = np.reshape(hog_features, hog_features.shape[0])
        filename = os.path.join(opt.output, os.path.basename(i).replace('.png', '.npy'))
        np.save(filename, hog_features)
list_of_features = get_list_of_files(opt.output, '*.npy')
kmeans = KMeans(n_clusters = opt.kmeans, random_state=0)
dict_of_hogs = dict()
for i in list_of_features:
    dict_of_hogs[i] = np.load(i)
list_of_hogs = list(dict_of_hogs.values())
list_of_hogs = np.asarray(list_of_hogs, dtype = object)
kmeans.fit(list_of_hogs)
clusters = kmeans.labels_
for c in zip(dict_of_hogs.keys(), clusters):
    filename = os.path.basename(c[0]).replace('.npy', '.png')
    src = os.path.join(opt.data, filename)
    dst = os.path.join(opt.clusters, str(c[1]), filename)
    os.link(src, dst)
