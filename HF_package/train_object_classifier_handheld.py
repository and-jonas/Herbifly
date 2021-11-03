
# ======================================================================================================================
# Author: Jonas Anderegg, jonas.anderegg@usys.ethz.ch
# Project: Herbifly;
# Date: 14.08.2020
# Sample training data for component classification; Train classifier
# ======================================================================================================================

# imports
import glob
import numpy as np
from pathlib import Path
import imageio
import matplotlib.image as mpimg
import os
import pandas as pd
import re
import random
import pickle
from sklearn.ensemble import RandomForestClassifier

from HF_package import ClfFunctions
from HF_package import utils

# ======================================================================================================================
# (1) SELECT SUBSET OF IMAGES
# ======================================================================================================================

work_dir = "O:/Evaluation/Hiwi/2020_Herbifly"

# paths
img_dir = Path(f'{work_dir}/Images_Farmers/')
output_dir = Path(f'{work_dir}/Images_Farmers/Output/Handheld_output/')

# select early time points
alldates = [os.path.basename(x) for x in glob.glob(f'{output_dir}/**/segmentation/*')]
earlydates = list(np.unique(alldates)[0:9])
all_files = glob.glob(f'{output_dir}/**/segmentation/**/*[0-9].tif')
subset_files = []
for i in all_files:
    name_elements = i.split('/')
    n_elements = len(name_elements)
    idx = n_elements-1
    date = name_elements[idx-1]
    if date in earlydates:
        subset_files.append(i)
    else:
        continue

# ======================================================================================================================
# (2) POST-PROCESS VEGETATION MASKS
# ======================================================================================================================

for file in subset_files:

    # get/set paths
    path = re.sub(".tif", "", file)
    nameidx = [file.split('/')[i] for i in [7, 9, 10]]
    nameidx.append(utils.get_farmer_region(nameidx[0]))
    nameidx[2] = re.sub(".tif", "", nameidx[2])
    trcname = nameidx[2]

    # sink directories
    savedir = [file.split('/')[i] for i in range(9)]
    savedir = "/".join(savedir)
    maskname = "".join([trcname, "_mask_pp.tif"])
    imgname = "".join([trcname, "_img_pp.tif"])
    maskname_rec = "".join([trcname, "_mask_pp_all_rec.tif"])
    date = nameidx[1]
    fullname_mask = f'{savedir}/{date}/{maskname}'
    fullname_img = f'{savedir}/{date}/{imgname}'
    fullname_mask_rec = f'{savedir}/{date}/{maskname_rec}'

    # load files
    try:
        img = mpimg.imread(str(f'{path}_tile_orig.tif'))
        mask = mpimg.imread(str(f'{path}.tif'))
    except FileNotFoundError:
        print("Some components not found. Skipping.")
        continue

    # post-process masks
    if Path(fullname_mask_rec).exists() and Path(fullname_mask).exists():
        print("Output already exists. Skipping post-processing.")
        continue
    else:
        img_cnts, img_cmask, mask_rec = ClfFunctions.post_process_hh_mask(img=img,
                                                                          mask=mask,
                                                                          min_size=250)
        imageio.imwrite(fullname_mask, img_cmask)
        imageio.imwrite(fullname_img, img_cnts)
        imageio.imwrite(fullname_mask_rec, mask_rec)

# ======================================================================================================================
# (3) GET TRAINING POSITIONS
# ======================================================================================================================

oontinue_proc = True  # Optionally, define stopping procedure
while oontinue_proc:

    # randomly select an image to use for training
    file = random.sample(subset_files, k=1)[0]

    # get/set paths
    path = re.sub(".tif", "", file)
    nameidx = [file.split('/')[i] for i in [7, 9, 10]]
    nameidx.append(utils.get_farmer_region(nameidx[0]))
    nameidx[2] = re.sub(".tif", "", nameidx[2])
    trcname = nameidx[2]

    # load required files
    try:
        img_cnts = mpimg.imread(str(f'{path}_img_pp.tif'))
        img_cmask = mpimg.imread(str(f'{path}_mask_pp.tif'))
        img_cmask_rec = mpimg.imread(str(f'{path}_mask_pp_all_rec.tif'))
    except FileNotFoundError:
        print("Some components not found. Skipping.")
        continue

    # sample training positions
    # close GUI if proposed image is inadequate
    # terminate console when finished
    # !! MAKE SURE COMPONENTS ARE NOT LABELLED MULTIPLE TIMES !!
    # !! MARKED POSITIONS MUST LIE WITHIN ERODED AND RECONSTRUCTED OBJECTS !!
    try:
        training_coords = []
        training_coords = ClfFunctions.capture_training_pos_GUI_obj_hh(img=img_cnts,
                                                                       pp_mask_rec=img_cmask_rec,
                                                                       pp_mask_er=img_cmask,
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
    path_out = f'{output_dir}/test_output_handheld/training_coords/'
    if not Path(path_out).exists():
        Path(path_out).mkdir(parents=True, exist_ok=True)
    filename = f'{path_out}/{trcname}.csv'
    if not Path(filename).exists():
        train.to_csv(f'{path_out}/{trcname}.csv', index=False)
    else:
        print("File already exists! Skipping...")
        continue

# ======================================================================================================================
# (4) EXTRACT FEATURES FOR PREVIOUSLY LABELLED OBJECTS
# ======================================================================================================================

# iterate over all images used for training
# extract features for each annotated object
path_to_files = Path(f'{output_dir}/test_output')
ClfFunctions.iterate_training_images(path_to_files=path_to_files,
                                     img_dir=img_dir,
                                     plot_pos=False,
                                     reconstruct=True)

ClfFunctions.iterate_training_images(path_to_files=path_to_files,
                                     img_dir=img_dir,
                                     plot_pos=False,
                                     reconstruct=False)

# ======================================================================================================================
# (5) TRAIN RANDOM FOREST CLASSIFIER
# ======================================================================================================================

# Prepare training data
df = pd.io.parsers.read_csv(f'{output_dir}/test_output/training_data/reconstruct/training_data_bal.csv')
# response vector
y = np.asarray(df.loc[:, df.columns == 'class_label'])
# predictor matrix
# # convert the character strings to a binary dummy variable
# ctype_dummies = pd.get_dummies(df[['ctype']])
# df['ctype_eroded'] = ctype_dummies['ctype_eroded']
# df['ctype_original'] = ctype_dummies['ctype_original']
df = df.drop(['ctype'], axis=1)
X = np.asarray(df.drop(['class_label'], axis=1))

# Specify model hyper-parameters (no hyper-parameter tuning)
clf = RandomForestClassifier(
    max_depth=95,  # maximum depth of 95 decision nodes for each decision tree
    max_features=11,  # maximum of 11 features are considered when forming a decision node
    min_samples_leaf=5,  # minimum of 5 samples needed to form a final leaf
    min_samples_split=4,  # minimum 4 trainings needed to create a decision node
    n_estimators=300,  # form 300 decision trees
    bootstrap=False,  # don't reuse samples
    random_state=1,
    n_jobs=-1
)

model = clf.fit(X, y.ravel())
score = model.score(X, y.ravel())
model.feature_importances_

# save model
path = 'O:/Evaluation/Hiwi/2020_Herbifly/Images_Farmers/Meta/classification_model/Handheld/'
pkl_filename = f'{path}/clf_comps_rec_rf.pkl'
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)

# ======================================================================================================================
