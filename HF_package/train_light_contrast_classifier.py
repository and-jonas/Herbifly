
# ======================================================================================================================
# Author: Jonas Anderegg, jonas.anderegg@usys.ethz.ch
# Project: Herbifly
# 13.05.2020
# Last modified 2021-02-11
# ======================================================================================================================

# imports
import glob
import matplotlib.image as mpimg
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pickle
import HF_package.LcClfUtils as LcClfUtils

# ======================================================================================================================

# (1) GET TRAINING DATA

picture_type = "10m"
wd = "O:/Evaluation/Hiwi/2020_Herbifly/Images_Farmers/Meta/light_contrast"
training_img_dir = f'{wd}/{picture_type}/training'

# hlc images
file_names = glob.glob(f'{training_img_dir}/hlc/*.JPG')

# get r,g,b histograms
xs1 = []
for filename in file_names:
    img = mpimg.imread(filename)
    hists = LcClfUtils.get_hist(img)
    xs1.append(hists)

# predictor matrix
X_1 = np.concatenate(xs1, axis=0)
# response vector
y_1 = np.repeat(1, len(X_1))

# llc images
file_names = glob.glob(f'{training_img_dir}/llc/*.JPG')

# get r,g,b histograms
xs2 = []
for filename in file_names:
    img = mpimg.imread(filename)
    hists = LcClfUtils.get_hist(img)
    xs2.append(hists)

# predictor matrix
X_2 = np.concatenate(xs2, axis=0)
# response vector
y_2 = np.repeat(2, len(X_2))

# assemble fulL training data
Xtrain = np.concatenate([X_1, X_2], axis=0)
ytrain = np.concatenate([y_1, y_2], axis=0)

# ======================================================================================================================

# (2) TRAIN SVM CLF

# 5-times repeated 10-fold CV for model evaluation
cv = RepeatedKFold(n_splits=10, n_repeats=5)

# check
for train, test in cv.split(Xtrain):
    print("%s %s" % (train, test))

# support vector machine with a radial basis function kernel,
# enable/disable class-membership probability estimation (involves internal 5-fold CV)
clf = SVC(kernel='rbf', verbose=1, probability=True)

# Data (n,p)
Xtrain.shape
ytrain.shape

# hyper-parameter grid
C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)

# assign to GridSearchCV
model = GridSearchCV(clf, param_grid=param_grid, cv=cv)

# fit the model
model.fit(Xtrain, ytrain)

# ======================================================================================================================

# (3) EVALUATE MODEL

# check result of hyper-parameter tuning
print("The best parameters are %s with a score of %0.2f"
      % (model.best_params_, model.best_score_))

scores = model.cv_results_['mean_test_score'].reshape(len(C_range), len(gamma_range))

# Save to file in the current working directory
pkl_filename = f'{wd}/{picture_type}/models/lc_svm.pkl'

with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)

# ======================================================================================================================

# (4) GET TEST DATA

# hlc images
validation_img_dir = f'{wd}/{picture_type}/validation'
file_names = glob.glob(f'{validation_img_dir}/hlc/**/*.jpg')

# get r,g,b histograms
xs1 = []
for filename in file_names:
    img = mpimg.imread(filename)
    hists = LcClfUtils.get_hist(img)
    xs1.append(hists)

# predictor matrix
X_1 = np.concatenate(xs1, axis=0)
# response vector
y_1 = np.repeat(1, len(X_1))

# llc images
file_names = glob.glob(f'{validation_img_dir}/llc/**/*.jpg')

xs2 = []
for filename in file_names:
    img = mpimg.imread(filename)
    hists = LcClfUtils.get_hist(img)
    xs2.append(hists)

# predictor matrix
X_2 = np.concatenate(xs2, axis=0)
# response vector
y_2 = np.repeat(2, len(X_2))

# assemble full test data
X_test = np.concatenate([X_1, X_2], axis=0)
y_test = np.concatenate([y_1, y_2], axis=0)

# ======================================================================================================================

# (5) TEST MODEL

# load from file
with open(pkl_filename, 'rb') as file:
    svm_clf = pickle.load(file)

# Calculate the accuracy score and predict target values
score = svm_clf.score(X_test, y_test)

probs = svm_clf.predict_proba(X_test)[:, 0]

dd = np.where(np.logical_and(probs <= 0.75, probs >= 0.25))  # border images pose problems!

# ======================================================================================================================
