# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 17:22:20 2019

@author: Tazrin
"""

import pandas as pd
import numpy as np
import math

def euclideanDistance(idx, edList):
    for i in range(len(data.index)):
        if i not in naIndex:
            n=0
            d0 = 0
            if featurePointer!=n:   #Hamming Distance for categorical feature
                if data.iloc[i,n]!=data.iloc[idx,n]:
                    d0 = 1
            n=n+1
            d1=0
            if featurePointer!=n:   #Euclidean Distance for continuous feature 1
                #print(data.iloc[i,n])
                #print(data.iloc[idx,n])
                d1 = pow((data.iloc[i,n]-data.iloc[idx,n]),2)
                d1 = math.sqrt(d1)
            n=n+1
            d2=0
            if featurePointer!=n:   #Euclidean Distance for continuous feature 2
                ##print(data.iloc[i,n])
                ##print(data.iloc[idx,n])
                d2 = pow((data.iloc[i,n]-data.iloc[idx,n]),2)
                ##print(d2)
                d2 = math.sqrt(d2)
                ##print(d2)
                #d2 = abs(data.iloc[i,n]-data.iloc[idx,n])
            distSum = d0+d1+d2
            ##print(distSum)
            edList.append([distSum, i])
            
def manhattanDistance(idx, edList):
    #print(idx)
    for i in range(len(data.index)):
        if i not in naIndex:
            n=0
            d0 = 0
            if featurePointer!=n:   #Hamming Distance for categorical feature
                if data.iloc[i,n]!=data.iloc[idx,n]:
                    d0 = 1
            n=n+1
            d1=0
            if featurePointer!=n:   #Manhattan Distance for continuous feature 1
                d1 = abs(data.iloc[i,n]-data.iloc[idx,n])
            n=n+1
            d2=0
            if featurePointer!=n:   #Manhattan Distance for continuous feature 2
                d2 = abs(data.iloc[i,n]-data.iloc[idx,n])
                #d2 = abs(data.iloc[i,n]-data.iloc[idx,n])
            distSum = d0+d1+d2
            edList.append([distSum, i])
            
def cosineDistance(idx, edList):
    #print(idx)
    for i in range(len(data.index)):
        if i not in naIndex:
            n=0
            d0 = 0
            if featurePointer!=n:   #Hamming Distance for categorical feature
                if data.iloc[i,n]!=data.iloc[idx,n]:
                    d0 = 1
            n=n+1
            a0 =0
            b0=0
            if featurePointer!=n:   #Cosine Distance for continuous feature 1
                a0 = data.iloc[i,n]
                b0 = data.iloc[idx,n]
            n=n+1
            a1=0
            b1 =0
            if featurePointer!=n:   #Cosine Distance for continuous feature 2
                a1 = data.iloc[i,n]
                b1 = data.iloc[idx,n]
                #d2 = abs(data.iloc[i,n]-data.iloc[idx,n])
            ASum = math.sqrt(a0*a0 + a1*a1)
            BSum = math.sqrt(b0*b0 + b1*b1)
            distSum = (((a0*a1)+(b0*b1))/(ASum*BSum))+d0
            edList.append([distSum, i])



def oneNN(dataset, distanceList):
    #print(distanceList[0][1])
    dataset.iloc[naidx,featurePointer] = dataset.iloc[distanceList[0][1],featurePointer]
    
def KNN(datasetKNN,distanceList):
    k=10
    if featurePointer==0:
        modeList= []
        for distIdx in Edistance[0:k]:
            modeList.append(datasetKNN.iloc[distIdx[1],featurePointer])
        datasetKNN.iloc[naidx,featurePointer] = max(set(modeList), key=modeList.count)
        
    else:
        
        sumIdx =0
        for distIdx in Edistance[0:k]:
            print(distIdx)
            sumIdx = sumIdx + datasetKNN.iloc[distIdx[1],featurePointer]
    
        datasetKNN.iloc[naidx,featurePointer] = sumIdx/k     #datasetKNN.iloc[minDistIdx,featurePointer]
    
def WKNN(datasetWKNN, distanceList):
    weight = []
    sumInvDist =0
    for j in distanceList[0:10]:
        #print(j[0])
        invDist =0
        if(j[0]==0):
            invDist = 0
        else:
            invDist = 1/j[0]
        weight.append([invDist, j[1]])
        sumInvDist = sumInvDist + invDist
        #sumInvDist = sum(weight[0])
    for k in weight:
        if sumInvDist!=0:
            k[0] = k[0]/sumInvDist
        
    minDist = max(weight)[1]
    #print(minDist)
    ##print(data.iloc[minDist,featurePointer])
    datasetWKNN.iloc[naidx,featurePointer] = datasetWKNN.iloc[minDist,featurePointer]

def numericalAccuracy(ds):
    difference = 0
    sumOriginal =0
    for i in naIndex:
        #print(i)
        #print(df.iloc[i,featurePointer])
        #print(ds.iloc[i,featurePointer])
        difference = difference + abs(df.iloc[i,featurePointer]-ds.iloc[i,featurePointer])
        sumOriginal = sumOriginal+df.iloc[i,featurePointer]
    accuracyNum = (1-(difference/sumOriginal))*100
    return accuracyNum

def categoricalAccuracy(ds):
    accurate = 0
    for i in naIndex:
        if(df.iloc[i,featurePointer]==ds.iloc[i,featurePointer]):
            accurate=accurate+1
    accuracyCat = (accurate/len(naIndex))*100
    return accuracyCat

#***********************************************************************************************#
accuracyList = []
accuracyList.append([" ", "Distance Measure", "1NN", "KNN", "Weighted KNN"])
#counter =0
#while(counter!=3):
featurePointer = 0
df = pd.read_csv('data.csv')
#print(counter)
#accuracyList.append([" ", " ", "20% ", " ", " "])
while(featurePointer!=3):
    data = pd.read_csv('data.csv')
    
    naIndex = list()
    #missing =0
    #if counter==0:
        #missing =0.05
    #elif counter == 1:
        #missing = 0.10
    #else:
    missing = 0.05 #Please change this value to 0.1 & 0.2 to see accuracy of 10% & 20% missing values respectively
    for i in data.sample(int(missing*len(data.index))).index:
        naIndex.append(i)
        data.iloc[i,featurePointer] = 'NA'
        
    #counter= counter+1
    dataset = data.copy()
    datasetKNN = data.copy()
    datasetWKNN = data.copy()
    #print(featurePointer)
    for naidx in naIndex:
    #Calculate distance of each null idx with others
        print(featurePointer)
        Edistance = list()
        euclideanDistance(naidx,Edistance)
        list.sort(Edistance)
        
        oneNN(dataset,Edistance)
        KNN(datasetKNN,Edistance)
        WKNN(datasetWKNN,Edistance)
        
    if featurePointer==0:
        #print('yes')
        accuracyList.append([data.columns[featurePointer],'Euclidean Distance',categoricalAccuracy(dataset),categoricalAccuracy(datasetKNN),categoricalAccuracy(datasetWKNN)])
    else:
        accuracyList.append([data.columns[featurePointer],'Euclidean Distance',numericalAccuracy(dataset),numericalAccuracy(datasetKNN),numericalAccuracy(datasetWKNN)])
    
    for naidx in naIndex:
    #Calculate distance of each null idx with others
        CDistance = list()
        cosineDistance(naidx,CDistance)
        list.sort(CDistance)
        
        oneNN(dataset,CDistance)
        KNN(datasetKNN,CDistance)
        WKNN(datasetWKNN,CDistance)
        
    #numericalAccuracy(dataset)
    #numericalAccuracy(datasetKNN)
    #numericalAccuracy(datasetWKNN)
    if featurePointer == 0:
        accuracyList.append([' ', 'Cosine Distance',categoricalAccuracy(dataset),categoricalAccuracy(datasetKNN),categoricalAccuracy(datasetWKNN)])
    else:
        accuracyList.append([' ', 'Cosine Distance',numericalAccuracy(dataset),numericalAccuracy(datasetKNN),numericalAccuracy(datasetWKNN)])
    featurePointer = featurePointer+1
    
    

import csv
with open('resultsFromCode.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(accuracyList)

print("Results are saved in resultsFromCode.csv file")