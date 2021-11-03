
# ======================================================================================================================
# Author: Jonas Anderegg, jonas.anderegg@usys.ethz.ch
# Project: Herbifly
# Date: 25.05.2020
# ======================================================================================================================

# imports
import numpy as np
import cv2
import pickle

# ======================================================================================================================


# Function to extract RGB intensity histograms for images
def get_hist(img):

    # extract histograms
    cols = ('r', 'g', 'b')
    hist = []
    # loop over color channels
    for i, col in enumerate(cols):
        # get intensity histogram
        histo = cv2.calcHist(img, [i], None, [256], [0, 256])
        # normalize histogram (so that image size does not matter for clf)
        norm_histo = np.true_divide(histo, histo.sum())
        hist.append(norm_histo)

    # get concatenated histograms in long format
    h_df = np.concatenate(hist, axis=0)
    out = np.transpose(h_df)

    return out


# Function to predict the light contrast class for images
def pred_lc(img, path_model):

    # if required, load pre-trained svm clf model
    if 'svm_clf' not in locals() and 'svm_clf' not in globals():
        print('loading model')
        with open(path_model, 'rb') as file:
            global svm_clf
            svm_clf = pickle.load(file)

    # get predictors for image (i.e. the concatenated r,g,b histograms in wide format) as a numpy array
    X = get_hist(img)

    # create a class label prediction using the pre-trained svm_clf
    pred_class_label = svm_clf.predict(X)[0]

    # get probability estimates
    prob_ests = svm_clf.predict_proba(X)

    return {"lab": pred_class_label,
            "prob_c1": prob_ests.flat[0],
            "prob_c2": prob_ests.flat[1]}


# Function to convert the binary class to one of three classes using class probabilities
def proc_lc_lab(svm_result, th=0.25):
    lc_lab = svm_result['lab']  # class label
    prob = svm_result['prob_c1']  # class probability
    # overwrite class label with class 3, if probability is below threshold
    if prob <= 1-th and prob >= th:
        lc_lab = 3
    return lc_lab


# ======================================================================================================================
