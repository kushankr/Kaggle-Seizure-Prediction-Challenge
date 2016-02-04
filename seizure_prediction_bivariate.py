import numpy as np
from scipy.io import loadmat
import glob
from sklearn.linear_model import LogisticRegression
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
        dataarray = np.array(data['data'][0][0])
        dfappend1 = []
        dfappend2 = []
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
            for item1 in count[item + 1:]:
                final = dataarray[item1]
                final = np.array(final)
                s2 = (final - np.mean(final)) / np.std(final)
                dfappend1.append(np.corrcoef(s1, s2)[0][1])
                dfappend2.append(np.linalg.eig(np.corrcoef(s1, s2))[0][0])
                dfappend2.append(np.linalg.eig(np.corrcoef(s1, s2))[0][1])
        dfappend = np.concatenate((dfappend1, dfappend2))
        testdata.append(dfappend)
        target.append(file)
    return testdata, target


def extract_features(files):
    finaldata = []
    target = []
    for file in glob.glob(files):
        if 'preictal' in file:
            targetvalue = 1
            mat = loadmat(file)
        if 'interictal' in file:
            targetvalue = 0
            mat = loadmat(file)
        if 'test' in file:
            continue
        for key in mat.keys():
            if 'preictal' in key:
                data = mat[key]
            if 'interictal' in key:
                data = mat[key]
        dataarray = np.array(data['data'][0][0])
        dfappend1 = []
        dfappend2 = []
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
            for item1 in count[item + 1:]:
                final = dataarray[item1]
                final = np.array(final)
                s2 = (final - np.mean(final)) / np.std(final)
                dfappend1.append(np.corrcoef(s1, s2)[0][1])
                dfappend2.append(np.linalg.eig(np.corrcoef(s1, s2))[0][0])
                dfappend2.append(np.linalg.eig(np.corrcoef(s1, s2))[0][1])
        dfappend = np.concatenate((dfappend1, dfappend2))
        finaldata.append(dfappend)
        target.append(targetvalue)
    return finaldata, target


folder_list = ["Dog_1", "Dog_2", "Dog_3", "Dog_4", "Dog_5", "Patient_1", "Patient_2"]
for folder in folder_list:
    X_train, y_train = extract_features(str(folder) + "\\*.mat")
    model = LogisticRegression(penalty='l2', class_weight="auto")
    model.fit(X_train, y_train)
    X_test, filename = extract_test_features(str(folder) + "\\*.mat")
    predictions = model.predict(X_test)
    filename = [f.replace(str(folder) + "\\", "") for f in filename]

    # https://www.kaggle.com/wiki/GettingStartedWithPythonForDataScience
    predicted_probs = [[str(filename[index]), float(round(x[1], 4))] for index, x in
                       enumerate(model.predict_proba(X_test))]

    savetxt(
        'Submission_' + str(folder) + '_v1.csv',
        predicted_probs,
        delimiter=',',
        header='clip,preictal',
        fmt="%s",
        comments=''
    )
