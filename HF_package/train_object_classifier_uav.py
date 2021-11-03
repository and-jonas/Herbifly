
# ======================================================================================================================
# Author: Jonas Anderegg, jonas.anderegg@usys.ethz.ch
# Project: Herbifly;
# Date: 14.08.2020
# Sample training data for component classification; Train classifier
# ======================================================================================================================

# imports
import glob
import os
import numpy as np
from pathlib import Path
import imageio
import matplotlib.image as mpimg
import pandas as pd
import re
import random
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from HF_package import utils
from HF_package import ClfFunctions

# ======================================================================================================================
# (1) SELECT IMAGES
# ======================================================================================================================

work_dir = "O:/Evaluation/Hiwi/2020_Herbifly"

# paths
img_dir = Path(os.path.join(work_dir, "Images_Farmers/"))
output_dir = Path(os.path.join(work_dir, 'Images_Farmers/Output/10m_output/'))
# list images in directories
all_files = glob.glob(f'{output_dir}/**/segmentation/**/*[0-9].tif')
# all_files = glob.glob(f'{work_dir}/Images_Farmers/Output/10m_output/test_output/training_coords/*.csv')

# ======================================================================================================================
# (2) GET TRAINING POSITIONS
# ======================================================================================================================

continue_proc = True
while continue_proc:

    # randomly select an image to use for training
    file = random.sample(all_files, k=1)[0]

    # get/set paths
    path = re.sub(".tif", "", file)
    nameidx = [file.split('/')[i] for i in [7, 9, 10]]
    nameidx.append(utils.get_farmer_region(nameidx[0]))
    nameidx[2] = re.sub(".tif", "", nameidx[2])
    trcname = "_".join(nameidx[0:3])

    # load required files
    try:
        img = mpimg.imread(str(f'{path}_tile_orig.tif'))
        mask = mpimg.imread(str(f'{path}.tif'))
    except FileNotFoundError:
        print("Some components not found. Skipping.")
        continue

    # post-process mask ("on-the-go" for uav-images)
    mask_pp = ClfFunctions.post_process_mask(mask)
    imageio.imwrite(f'{path}_mask_pp.tif', mask_pp)

    # sample training positions
    # close GUI if proposed image is inadequate
    # terminate console when finished
    # !! MAKE SURE COMPONENTS ARE NOT LABELLED MULTIPLE TIMES !!
    # !! MARKED POSITIONS MUST LIE WITHIN OBJECTS !!
    try:
        training_coords = []
        training_coords = ClfFunctions.capture_training_positions_GUI_objects(img=img,
                                                                              mask=mask_pp,
                                                                              training_coordinates=training_coords)
        if not training_coords:
            raise ValueError('image not used, loading next')
    except ValueError:
        print('image not used, loading next')
        continue

    # add image path to positions
    for cs in training_coords:
        cs.update({"img_id": file})

    # save training coordinates to .csv
    train = pd.DataFrame(training_coords)
    path_out = f'{output_dir}/test_output/training_coords'
    if not Path(path).exists():
        Path(path).mkdir(parents=True, exist_ok=True)
    filename = f'{output_dir}/{trcname}.csv'
    if not Path(filename).exists():
        train.to_csv(f'{path_out}/{trcname}.csv', index=False)
    else:
        print("File already exists! Skipping...")
        continue

# ======================================================================================================================
# (3) EXTRACT FEATURES FOR PREVIOUSLY LABELLED OBJECTS
# ======================================================================================================================

path_to_files = Path(f'{output_dir}/test_output')
ClfFunctions.iterate_training_images_uav(path_to_files=path_to_files,
                                         img_dir=img_dir,
                                         plot_pos=True,
                                         features="all")
ClfFunctions.iterate_training_images_uav(path_to_files=path_to_files,
                                         img_dir=img_dir,
                                         plot_pos=False,
                                         features="subset")

# ======================================================================================================================
# (4) TRAIN RANDOM FOREST CLASSIFIER
# ======================================================================================================================

# REFACTOR FUNCTION FOR SUBSET
# THEN CONTINUE HERE

df = pd.io.parsers.read_csv(f"{work_dir}/Images_Farmers/Output/10m_output/test_output/training_data/all/training_data_bal.csv")

X = np.asarray(df[['mean_row_tgi', 'mean_is_tgi', 'tgi_ctr']])
y = np.asarray(df.loc[:, df.columns == 'class_label'])

logreg = LogisticRegression()
logreg.fit(X, y.ravel())

logreg.predict(X)

# save model
path = f'{work_dir}/Images_Farmers/Meta/classification_model'
pkl_filename = f'{path}/clf_comps_logit.pkl'
with open(pkl_filename, 'wb') as file:
    pickle.dump(logreg, file)

# ======================================================================================================================

# Specify model hyper-parameters
clf = RandomForestClassifier(
    max_depth=95,  # maximum depth of 95 decision nodes for each decision tree
    max_features=6,  # maximum of 6 features (channels) are considered when forming a decision node
    min_samples_leaf=6,  # minimum of 6 trainings needed to form a final leaf
    min_samples_split=4,  # minimums 4 trainings needed to create a decision node
    n_estimators=55,  # maximum of 55 decision trees
    bootstrap=False,  # don't reuse trainings
    random_state=1,
    n_jobs=-1
)

X = np.asarray(df.drop(['class_label'], axis=1))
y = np.asarray(df.loc[:, df.columns == 'class_label'])

model = clf.fit(X, y.ravel())
score = model.score(X, y)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# model = clf.fit(X_train, y_train.ravel())
# score = model.score(X, y)
#
# y_pred = model.predict(X_test)
# print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(model.score(X_test, y_test)))

# save model
path = f'{work_dir}/Images_Farmers/Meta/classification_model'
pkl_filename = f'{path}/clf_comps_rf.pkl'
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)

# ======================================================================================================================
