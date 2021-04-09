#!/usr/bin/env python
# -*- coding: latin-1 -*-
import cv2
import os
import sys
import numpy as np

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
    print centroids
            
def main():
    img = cv2.imread(sys.argv[1])
    get_centroids(8, img, 1.0, 5.0)
        

if __name__ == '__main__':
    main()
