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
    #print(centroids)
    while len(centroids) < m:
        centroids = disturb_centroids(centroids, epsilon)
        #print("Calculating " + str(len(centroids)) + " centroids")
        delta = tol + 1
        while delta > tol:
            clusters = [[] for c in centroids]
            for p in img:
                idx = np.argmin([np.linalg.norm(p-c) for c in centroids])
                clusters[idx].append(p)
            new_centroids = [np.mean(c, axis=0) for c in clusters]
            delta = np.sum([np.linalg.norm(new_centroids[i] - centroids[i]) for i in range(len(centroids))])
            centroids = new_centroids
            #print("Current delta: " + str(delta))
    return centroids
 
def clr():
    print("\033[2J\033[1;1f")   
            
def main():
    pattern = sys.argv[1];
    img_files = glob.glob("dataset/" + pattern + "/*.jpg")    
    imgs = []
    
    delta = 0
    a = "Files to check: " + str(len(img_files))
    print(a)
    for file in img_files:
        imgs.append(cv2.imread(file))
        
    print("Reading file centroids:")
    npz_files = glob.glob("pypatterns/*.npz")
    a = "number of npz files: " + str(len(npz_files))
    print(a)
    D = [[] for l in npz_files]    
        
    for i, im in enumerate(imgs):
        img = images_to_vector(im)
        centroids = get_centroids(8, img, 1.0, 5.0)        
        
        for j, file in enumerate(npz_files):
            data = np.load(file)
            lcentroids = data["centroids"]
            
            for k in range(len(lcentroids)):
                delta = delta + np.linalg.norm(centroids[k]-lcentroids[k])
            D[j] = delta
            delta = 0
        minindex = np.argmin(D)
        clr()
        cv2.destroyAllWindows()
        cv2.imshow(img_files[i],imgs[i])
        cv2.waitKey(200)
        for l in range(len(D)):
            a = "D[" + str(l) + "] " + "= " + str(D[l]) + " " + npz_files[l]
            print(a)
        b = "minindex: " + str(minindex)
        print(b)
        c = "Identificado como: " + str(npz_files[minindex])
        print(c)                

def images_to_vector(img):
    pixels = []
    rows, cols, dim = img.shape
    for i in range(rows):
        for j in range(cols):
            pixels.append(img[i][j])    
    return pixels

if __name__ == '__main__':
    main()
