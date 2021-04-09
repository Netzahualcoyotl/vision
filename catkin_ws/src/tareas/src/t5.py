#!/usr/bin/env python
# -*- coding: latin-1 -*-
import cv2
import os
import sys
import numpy as np
import glob
#import scipy as sc
#import matplotlib.pyplot as plt
#import tensorflow as tf


# CLASE DE LA CAPA DE LA RED
class neural_layer():
    def __init__(self, n_conn, n_neur, act_f):
        self.act_f = act_f
        self.b = np.random.rand(1, n_neur)*2 -1
        self.W = np.random.rand(n_conn, n_neur)*2 -1

#FUNCIONES DE ACTIVACION

sigm = (lambda x: 1 / (1 + np.e ** (-x)),
        lambda x: x * (1 - x))

relu = lambda x: np.maximum(0, x)

n = 5
p = 3
#X = np.array([[-1 ,1], [-1, 1]])
#Y = np.array([1 , 2])

def images_to_vector(img):
    pixels = []
    k = 0
    for image in img:
        rows, cols, dim = image.shape
        for i in range(rows):
            for j in range(cols):
                pixels.append(img[k][i][j])
        k = k + 1    
    return pixels


pattern = sys.argv[1];
img_files = glob.glob("dataset/training/" + pattern + "/*.jpg")
imgs = []
for file in img_files:
    imgs.append(cv2.imread(file))    
    img = images_to_vector(imgs)   

X = np.array(img)
#res = 3
#_x0 = np.linspace(-1.5, 1.5, res)
#_x1 = np.linspace(-1.5, 1.5, res)
#Y = np.zeros((res))




#for i0, x0 in enumerate(_x0):
#    for i1, x1 in enumerate(_x1):
#        Y[i0] = x0 ** 2 + x1 **2
#_x0 = np.linspace(0, 1, res)
#_x1 = np.linspace(0, 1, res)        
#X = np.array([_x0, _x1]).T
#X = X[:, np.newaxis]
#X.reshape(-1,2,0)
#X = np.array([[0, 0],
#              [0 ,1],
#              [1, 0],
#              [1, 1]])
print(X)
print(X.shape)
print(X)
#Y = np.array([[1],
#              [0],
#              [1],
#              [1]])
Y = np.zeros(len(X))


data = np.load("pypatterns/" + pattern + ".npz")
centroids = data["centroids"]


for i in range(len(centroids)):
    for j in range(len(X)):
        if np.linalg.norm(centroids[i]) != 0 and np.linalg.norm(X[j] != 0):
            Y[i] = (X[j][0]*centroids[i][0] + X[j][1]*centroids[i][1] + X[j][2]*centroids[i][2])/(np.linalg.norm(X[j])*(np.linalg.norm(centroids[i])))


print(Y.shape)
Y = Y[:, np.newaxis]
print(Y.shape)
print(Y)
#_x = np.linspace(-5, 5, 100)
#print(sigm[0](_x))
#print(sigm[1](_x))

#l0 = neural_layer(p, 4, sigm)
#l1 = neural_layer(4, 8, sigm)



def create_nn(topology, act_f):
    nn = []
    for l, layer in enumerate(topology[:-1]):
        nn.append(neural_layer(topology[l],topology[l+1],act_f))
        
    return nn

topology = [p, 4, 8, 1]

#neural_net = create_nn(topology, sigm)

l2_cost = (lambda Yp, Yr: np.mean((Yp -Yr) ** 2),
           lambda Yp, Yr: (Yp - Yr))



def train(neural_net, X, Y, l2_cost, lr = 0.5, train = True):

    out = [(None, X)]
    #FORWARD PASS
    for l, layer in enumerate(neural_net):
        #z = X @ neural_net[0].W + neural_net[0].b
        #a = neural_net[0].act_f(z)
        z = np.dot(out[-1][1], neural_net[l].W) + neural_net[l].b
        a = neural_net[l].act_f[0](z)
        out.append((z, a))
    #print(out[-1][1])
    #print(l2_cost[0](out[-1][1],Y))
    
    
    if train:
        #Backward pass
        deltas = []
        for l in reversed(range(0, len(neural_net))):
            
            # Calcular delta ultima capa
            z = out[l+1][0]
            a = out[l+1][1]
            #print(a.shape)
            if l == len(neural_net) -1:
                deltas.insert(0, l2_cost[1](a, Y) * neural_net[l].act_f[1](a))
            else:                                
                #print(deltas[0].T.shape)
                #print(_W.T.shape)
                deltas.insert(0, np.dot(deltas[0], _W.T) * neural_net[l].act_f[1](a))
                
            _W = neural_net[l].W
            
            # Calcular delta respecto a capa previa    
        
            # Gradient descent
            neural_net[l].b = neural_net[l].b - np.mean(deltas[0], axis = 0, keepdims = True) * lr
            neural_net[l].W = neural_net[l].W - np.dot(out[l][1].T, deltas[0]) * lr
        return out[-1][1]
    
    
#train(neural_net, X, Y, l2_cost, 0.5, True)   
    


#test
 
neural_n = create_nn(topology, sigm)

loss = []

for i in range(10000):
    #Entrenamiento
    pY = train(neural_n, X, Y, l2_cost, 0.5, True)
    if i % 25 == 0:
        loss.append(l2_cost[0](pY, Y))

print(pY) 
#print(loss)  
NN = [[]]
for i, nn in enumerate(neural_n):
    NN.append([nn.W, nn.b])
    
np.savez("pypatternsnn/" + pattern, NN = NN)   
    
    
    
    
    
    
