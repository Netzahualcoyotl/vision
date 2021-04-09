#!/usr/bin/env python
# -*- coding: latin-1 -*-
import cv2
import os
import sys
import numpy as np
import glob

def disturb_centroids(centroids, epsilon):
    new_centroids = []
    for c in centroids:
        v = np.random.rand(3)
        v = epsilon * v / np.linalg.norm(v)
        new_centroids.append(c + v)
        new_centroids.append(c - v)
    return new_centroids

def get_centroids(m, img, epsilon, tol):
    img = np.reshape(img, (-1,3))   
    centroids = [np.mean(img, axis=0)]
    print(centroids)
    while len(centroids) < m:
        centroids = disturb_centroids(centroids, epsilon)
        print("Calculating " + str(len(centroids)) + " centroids")
        delta = tol + 1
        while delta > tol:
            clusters = [[] for c in centroids]
            for p in img:
                idx = np.argmin([np.linalg.norm(p-c) for c in centroids])
                clusters[idx].append(p)
            new_centroids = [np.mean(c, axis=0) for c in clusters]
            delta = np.sum([np.linalg.norm(new_centroids[i] - centroids[i]) for i in range(len(centroids))])
            centroids = new_centroids
            print("Current delta: " + str(delta))
    return centroids
    
            
def main():
    pattern = sys.argv[1];
    img_files = glob.glob("dataset/training/" + pattern + "/*.jpg")
    imgs = []
    for file in img_files:
        imgs.append(cv2.imread(file))
    
    img = images_to_vector(imgs)   
    centroids = get_centroids(8, img, 1.0, 5.0)
    np.savez("pypatterns/" + pattern,centroids = centroids)
    data = np.load("pypatterns/" + pattern + ".npz")
    centroids = data["centroids"]
    print(centroids)

def images_to_vector(img):
    pixels = []
    #k = 0
    for k, image in enumerate(img):
        rows, cols, dim = image.shape
        for i in range(rows):
            for j in range(cols):
                pixels.append(img[k][i][j])
        #k = k + 1    
    return pixels

if __name__ == '__main__':
    main()
