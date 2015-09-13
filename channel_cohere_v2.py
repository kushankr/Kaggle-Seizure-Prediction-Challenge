import matplotlib.pyplot as plt
import matplotlib.mlab as ml
import numpy as np
from scipy.io import loadmat 
import pandas as pd
import glob
from sklearn import svm
from numpy.fft import fft,fftfreq,rfft
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
import networkx as nx

compare = lambda x, y: collections.Counter(x) == collections.Counter(y)
#Source:http://rosettacode.org/wiki/Entropy
def entropy(s):
 p, lns = Counter(s), float(len(s))
 return -sum( count/lns * math.log(count/lns, 2) for count in p.values())
 
def calculate_correlation(array1):
 df_corr=[]
 i = len(array1);
 count=[]
 for j in range(i):
   count.append(j);
 initial =[] 
 for item in count:
   initial = array1[item]
   initial = np.array(initial)
   for item1 in count[item+1:]:
     final = array1[item1]
     final = np.array(final)
     df_corr.append(np.corrcoef(initial,final)[0][1])
 return df_corr
 
def calculate_correlation_eigen(array1):
 df_corr_eig=[]
 i = len(array1);
 count=[]
 for j in range(i):
   count.append(j);
 initial =[] 
 for item in count:
   initial = array1[item]
   initial = np.array(initial)
   for item1 in count[item+1:]:
     final = array1[item1]
     final = np.array(final)
     df_corr_eig.append(np.linalg.eig(np.corrcoef(initial,final))[0][0])
     df_corr_eig.append(np.linalg.eig(np.corrcoef(initial,final))[0][1])
 return df_corr_eig
 
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
  dataarraym = np.split(dataarray, 3,axis=1)
  for item in dataarraym:
   dfappend1=[];
   dfappend2=[];
   df=[];
   dataarray=item
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
    df.append(s1);
   dfappend1 = calculate_correlation(df)
   dfappend2 = calculate_correlation_eigen(df)   
   dfappend1 = preprocessing.scale(np.array(dfappend1).astype(float),axis=0)
   dfappend2 = preprocessing.scale(np.array(dfappend2).astype(float),axis=0)
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
   targetvalue = -1;
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
  dataarraym = np.split(dataarray, 3,axis=1)
  for item in dataarraym:
   dfappend1=[];
   dfappend2=[];
   df=[];
   dataarray=item
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
    df.append(s1);
   dfappend1 = calculate_correlation(df)
   dfappend2 = calculate_correlation_eigen(df)      
   dfappend1 = preprocessing.scale(np.array(dfappend1).astype(float),axis=0)
   dfappend2 = preprocessing.scale(np.array(dfappend2).astype(float),axis=0)
   dfappend = np.concatenate((dfappend1,dfappend2))
   finaldata.append(dfappend)
   target.append(targetvalue);
 return finaldata, target;
  
  
folder =["Dog_1","Dog_2","Dog_3","Dog_4","Dog_5","Patient_1","Patient_2"]
for fo in folder:
 X_train, y_train = extract_features(str(fo)+"\\*.mat")
 X_test,filename = extract_test_features(str(fo)+"\\*.mat")
 carr=[50]
 for cval in carr:
  clf = svm.SVC(C=cval,kernel="rbf",class_weight="auto")
  clf.fit(X_train,y_train)  
  predictions = clf.predict(X_test)
  filename = [f.replace(str(fo)+"\\","") for f in filename]
  predicted_probs=[]
  i = 0
  for pr in predictions:
   predicted_probs.append((str(filename[i]),float(pr)));
   i = i+1;
  pr_probs=[]
  for i,k,l in zip(predicted_probs[0::3], predicted_probs[1::3], predicted_probs[2::3]):
    print i[1],k[1],l[1]
    if i[1] == 1 and l[1] == 1 or i[1] == 1 and k[1] ==1 or k[1] == 1 and l[1] == 1 or i[1] == 1 and k[1] == 1 and l[1] == 1:
     print 'yes'
     pr_probs.append((i[0],1.0));
     continue;
    else:
     print 'no'
     pr_probs.append((i[0],0.0));
     continue;
  from numpy import genfromtxt, savetxt
  savetxt('Submission_'+str(fo)+'_'+str(cval)+'_v2.csv', pr_probs, delimiter=',',header='clip,preictal', fmt="%s", comments = '')



 





