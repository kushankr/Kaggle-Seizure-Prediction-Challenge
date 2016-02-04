Kaggle Seizure Prediction Challenge
-------------------------------------
#### Summary

Epilepsy affects 1% of world’s population
and is marked by spontaneous occurrence of seizures. There is no available reliable seizure prediction methods. A seizure prediction algorithm is proposed that will classify an Intracranial EEG segment as interictal (between seizures) or preictal (prior to seizure). I focused on computing linear bivariate features such as correlation coefficient between all pairs of channels and Eigen values of correlation coefficient matrix between all pairs of channels and linear uni-variate features such as power in eight different frequency bands. I employed L2-regularized Logistic regression and Support vector machines with non-linear kernels to make predictions. In certain cases, a 10 minute window was divided into 3 equal segments, each totaling 200 seconds. My best machine learning model, when applied to
EEG recordings of 7 subjects in the Kaggle’s
American Epilepsy Seizure Prediction Challenge database, produced an area under the ROC curve (AUC) score of 0.72690. This algorithm can be run on an implantable device and it can predict a seizure with moderate predictive accuracy. 

[Project Report](/docs/SeizurePrediction.pdf)
