# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import load_digits
digit=load_digits()

w=np.zeros((64,10))
v=np.zeros((64,10)) #(pixels,class)
net=np.zeros(10)
phi=np.zeros((64,1))

for k in range(10):
    w[0][k-1]=1
    v[0][k-1]=1

N=digit.data.shape[0] #the number of data=1797

for t in range(5*N):
    i=np.random.randint(1, N-50)
    
    phi=digit.data[i].reshape((64,1)) #ith feature vector
    
    for j in range(10):
        net[j]=phi.T.dot(w[:,j])
    
    max_i=np.argmax(net)
    
    if max_i!=digit.target[i]:
        w[:,digit.target[i]]+=phi[:,0]
        w[:,max_i]-=phi[:,0]
      
correct=0
        
for test in range(50):
    
    phi=digit.data[N-1-test].reshape((64,1)) #i番目の特徴ベクトル
    
    for j in range(10):
        net[j]=phi.T.dot(w[:,j])
    
    max_test=np.argmax(net)
    
    if max_test==digit.target[N-1-test]:
        correct+=1

print(correct/50)