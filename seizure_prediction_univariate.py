import numpy as np
from scipy.io import loadmat
import glob
from sklearn import svm
# Source: http://www.hindawi.com/journals/cin/2011/406391/
# https://code.google.com/p/pyeeg/
# Forrest S. Bao, Xin Liu and Christina Zhang,
# "PyEEG: An Open Source Python Module for EEG/MEG Feature Extraction,"
# Computational Intelligence and Neuroscience, March, 2011
import pyeeg
from sklearn import preprocessing
from numpy import savetxt


def extract_test_features(files):
    testdata = []
    target = []
    for file in glob.glob(files):
        if 'preictal' in file:
            continue
        if 'interictal' in file:
            continue
        if 'test' in file:
            mat = loadmat(file)
        for key in mat.keys():
            if 'test' in key:
                data = mat[key]
        Fs = data['sampling_frequency'][0][0][0][0]
        dataarray = np.array(data['data'][0][0])
        dataarraym = np.split(dataarray, 3, axis=1)
        for item in dataarraym:
            dfappend1 = []
            dataarray = item
            i = len(dataarray)
            count = []
            for j in range(i):
                count.append(j)
            initial = []
            i = 0
            for item in count:
                initial = dataarray[item]
                initial = np.array(initial)
                s1 = (initial - np.mean(initial)) / np.std(initial)
                p, pr = pyeeg.bin_power(
                    s1,
                    [0.5, 4, 8, 13, 30, 50, 70, 90, Fs / 2],
                    Fs
                )
                for it in p:
                    dfappend1.append(it)
                for item1 in count[item + 1:]:
                    final = dataarray[item1]
                    final = np.array(final)
            dfappend1 = preprocessing.scale(
                np.array(dfappend1).astype(float),
                axis=0
                )
            dfappend = dfappend1
            testdata.append(dfappend)
            target.append(file)
    return testdata, target


def extract_train_features(files):
    traindata = []
    target = []
    for file in glob.glob(files):
        if 'preictal' in file:
            targetvalue = 1
            mat = loadmat(file)
        if 'interictal' in file:
            targetvalue = -1
            mat = loadmat(file)
        if 'test' in file:
            continue
        for key in mat.keys():
            if 'preictal' in key:
                data = mat[key]
            if 'interictal' in key:
                data = mat[key]
        Fs = data['sampling_frequency'][0][0][0][0]
        dataarray = np.array(data['data'][0][0])
        dataarraym = np.split(dataarray, 3, axis=1)
        for item in dataarraym:
            dfappend1 = []
            dataarray = item
            i = len(dataarray)
            count = []
            for j in range(i):
                count.append(j)
            for item in count:
                initial = dataarray[item]
                initial = np.array(initial)
                s1 = (initial - np.mean(initial)) / np.std(initial)
                p, pr = pyeeg.bin_power(
                    s1,
                    [0.5, 4, 8, 13, 30, 50, 70, 90, Fs / 2],
                    Fs
                    )
                for it in p:
                    dfappend1.append(it)
                for item1 in count[item + 1:]:
                    final = dataarray[item1]
                    final = np.array(final)
            dfappend1 = preprocessing.scale(
                np.array(dfappend1).astype(float),
                axis=0
                )
            dfappend = dfappend1
            traindata.append(dfappend)
            target.append(targetvalue)
    return traindata, target


folder_list = ["Dog_1", "Dog_2", "Dog_3", "Dog_4", "Dog_5", "Patient_1", "Patient_2"]
for folder in folder_list:
    X_train, y_train = extract_train_features(str(folder) + "\\*.mat")
    clf = svm.SVC(C=50.0, kernel="poly", degree=4, class_weight="auto")
    clf.fit(X_train, y_train)
    X_test, filename = extract_test_features(str(folder) + "\\*.mat")
    predictions = clf.predict(X_test)
    filename = [f.replace(str(folder) + "\\", "") for f in filename]

    predicted_probs = []
    i = 0
    for pr in predictions:
        predicted_probs.append((str(filename[i]), float(pr)))
        i = i + 1
    pr_probs = []
    for i, k, l in zip(predicted_probs[0::3], predicted_probs[1::3], predicted_probs[2::3]):
        if i[1] == 1 and l[1] == 1 or i[1] == 1 and k[1] == 1 or k[1] == 1 and l[1] == 1 or i[1] == 1 and k[1] == 1 and l[1] == 1:
            pr_probs.append((i[0], 1.0))
            continue
        else:
            pr_probs.append((i[0], 0.0))
            continue

    savetxt(
        'Submission_' + str(folder) + '_v2.csv',
        pr_probs, delimiter=',',
        header='clip,preictal',
        fmt="%s",
        comments=''
        )
