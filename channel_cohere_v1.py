import matplotlib.pyplot as plt
import matplotlib.mlab as ml
import numpy as np
from scipy.io import loadmat 
import pandas as pd
import glob
from sklearn import svm
from numpy.fft import fft
from numpy.fft import ifft
from sklearn.metrics import roc_auc_score
import math
from collections import Counter
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import pywt
#from scipy.fftpack import fft
from scipy.fftpack import hilbert
from sklearn import linear_model
from sklearn import metrics
from math import e
from sklearn.metrics import accuracy_score
from sklearn import manifold
#Source: http://www.hindawi.com/journals/cin/2011/406391/
#https://code.google.com/p/pyeeg/
#Forrest S. Bao, Xin Liu and Christina Zhang, "PyEEG: An Open Source Python Module for EEG/MEG Feature Extraction," Computational Intelligence and Neuroscience, March, 2011 
import pyeeg
from sklearn.decomposition import PCA
import scipy.stats as st
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import collections
from scipy import signal
from tempfile import TemporaryFile
from numpy import matrix
import mlpy
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn import preprocessing

compare = lambda x, y: collections.Counter(x) == collections.Counter(y)
#Source:http://rosettacode.org/wiki/Entropy
def entropy(s):
 p, lns = Counter(s), float(len(s))
 return -sum( count/lns * math.log(count/lns, 2) for count in p.values())

def extract_test_features(files):
 testdata=[]
 target=[]
 for file in glob.glob(files):  
  if 'preictal' in file:
   print file
   continue;
  if 'interictal' in file:
   print file
   continue;
  if 'test' in file:
   print file
   mat = loadmat(file)
  #print mat.keys();
  for key in mat.keys():
    if 'test' in key:
      print key
      data = mat[key]
  Fs =  data['sampling_frequency'][0][0][0][0]
  dataarray= np.array(data['data'][0][0])
  dfappend1=[];
  dfappend2=[];
  dfappend3=[];
  dfappend4=[];
  i = len(dataarray);
  count=[]
  for j in range(i):
    count.append(j);
  initial =[]
  i = 0;  
  for item in count:
    initial = dataarray[item]
    initial = np.array(initial)
    s1 = (initial-np.mean(initial))/np.std(initial)
    df=[]
    df1=[]
    #N = 512
    #finitial = fft(initial, N)
    #finitial = np.array(abs(finitial))
    #ifinitial = ifft(finitial[:N/2],N)
    #hinitial = ifinitial
    #arr1=[]
    #for f in hinitial:
     #arr1.append(math.atan2(f.imag,f.real))
    for item1 in count[item+1:]:
      final = dataarray[item1]
      final = np.array(final)
      s2 = (final-np.mean(final))/np.std(final)
      dfappend1.append(np.corrcoef(s1,s2)[0][1])
      dfappend2.append(np.linalg.eig(np.corrcoef(s1,s2))[0][0])
      dfappend2.append(np.linalg.eig(np.corrcoef(s1,s2))[0][1])        
      #N = 512
      #ffinal = fft(final, N)
      #ffinal = np.array(abs(ffinal))
      #iffinal = ifft(ffinal[:N/2],N)
      #hfinal = iffinal
      #arr2=[]
      #for f in hfinal:
       #arr2.append(math.atan2(f.imag,f.real))
      #RP = [x - y for x, y in zip(arr1, arr2)]
      #arr=[]
      #for item in RP:
       #arr.append(e **(item*1j))
      #PLV =  abs(sum(arr))/len(RP)
      #dfappend.append(PLV)
  #dfappend1 = preprocessing.scale(np.array(dfappend1).astype(float),axis=0)
  #dfappend2 = preprocessing.scale(np.array(dfappend2).astype(float),axis=0)
  dfappend = np.concatenate((dfappend1,dfappend2))
  testdata.append(dfappend)
  target.append(file);
 return testdata, target;


def extract_features(files):
 finaldata=[]
 target=[]
 for file in glob.glob(files):  
  if 'preictal' in file:
   print file
   targetvalue = 1;
   mat = loadmat(file)
  if 'interictal' in file:
   print file
   targetvalue = 0;
   mat = loadmat(file)
  if 'test' in file:
   print file
   continue;
  #print mat.keys();
  for key in mat.keys():
    if 'preictal' in key:
      print key
      data = mat[key]
    if 'interictal' in key:
      print key
      data = mat[key]
  Fs =  data['sampling_frequency'][0][0][0][0]
  dataarray= np.array(data['data'][0][0])
  dfappend1=[];
  dfappend2=[];
  dfappend3=[];
  dfappend4=[];
  i = len(dataarray);
  count=[]
  for j in range(i):
    count.append(j);
  initial =[]
  i = 0;  
  for item in count:
    initial = dataarray[item]
    initial = np.array(initial)
    s1 = (initial-np.mean(initial))/np.std(initial)
    df=[]
    df1=[]
    #N = 512
    #finitial = fft(initial, N)
    #finitial = np.array(abs(finitial))
    #ifinitial = ifft(finitial[:N/2],N)
    #hinitial = ifinitial
    #arr1=[]
    #for f in hinitial:
     #arr1.append(math.atan2(f.imag,f.real))
    for item1 in count[item+1:]:
      final = dataarray[item1]
      final = np.array(final)
      s2 = (final-np.mean(final))/np.std(final)
      dfappend1.append(np.corrcoef(s1,s2)[0][1])
      dfappend2.append(np.linalg.eig(np.corrcoef(s1,s2))[0][0])
      dfappend2.append(np.linalg.eig(np.corrcoef(s1,s2))[0][1])        
      #N = 512
      #ffinal = fft(final, N)
      #ffinal = np.array(abs(ffinal))
      #iffinal = ifft(ffinal[:N/2],N)
      #hfinal = iffinal
      #arr2=[]
      #for f in hfinal:
       #arr2.append(math.atan2(f.imag,f.real))
      #RP = [x - y for x, y in zip(arr1, arr2)]
      #arr=[]
      #for item in RP:
       #arr.append(e **(item*1j))
      #PLV =  abs(sum(arr))/len(RP)
      #dfappend.append(PLV)
  #dfappend1 = preprocessing.scale(np.array(dfappend1).astype(float),axis=0)
  #dfappend2 = preprocessing.scale(np.array(dfappend2).astype(float),axis=0)
  dfappend = np.concatenate((dfappend1,dfappend2))
  finaldata.append(dfappend)
  target.append(targetvalue);
 return finaldata, target;
  
folder =["Dog_1","Dog_2","Dog_3","Dog_4","Dog_5","Patient_1","Patient_2"]
for fo in folder:  
 X_train, y_train = extract_features(str(fo)+"\\*.mat")
 model = LogisticRegression(penalty='l2',class_weight="auto")
 model.fit(X_train,y_train)
 X_test,filename = extract_test_features(str(fo)+"\\*.mat")
 predictions = model.predict(X_test)
 filename = [f.replace(str(fo)+"\\","") for f in filename]
 
 #https://www.kaggle.com/wiki/GettingStartedWithPythonForDataScience
 predicted_probs = [[str(filename[index]),float(round(x[1],4))] for index, x in enumerate(model.predict_proba(X_test))]
 from numpy import genfromtxt, savetxt
 savetxt('Submission_'+str(fo)+'_v1.csv', predicted_probs, delimiter=',',header='clip,preictal', fmt="%s", comments = '')





