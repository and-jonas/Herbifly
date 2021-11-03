
# ======================================================================================================================
# Author: Jonas Anderegg, jonas.anderegg@usys.ethz.ch
# Project: Herbifly
# 20.05.2020
# ======================================================================================================================

# imports
import glob
from pathlib import Path
# from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import os
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import imageio
import pickle

import HF_package.ImageCalculations as ImgCalc
import HF_package.LcClfUtils as ClassificationFunctions
import HF_package.ImageFunctions as ImageFunctions

# import matplotlib as mpl
# mpl.use('Qt5Agg')

# ======================================================================================================================
# 1. GET TRAINING DATA
# ======================================================================================================================

workdir = Path('./').resolve()

picture_type = "Handheld"

img_dir = Path('O:/Hiwi/2020_Herbifly/Images_Farmers')

all_files = glob.glob((f'{img_dir}/**/**/{picture_type}/**/*.JPG'))
len(all_files)  # 17036 potential training images

# files_to_process = random.sample(all_files)

# register number of images processed per lc class
count1 = 55
count2 = 7
count3 = 13
n_img = 15  # number of training images to use per class
continue_proc = True
while(continue_proc):
    # randomly select an image to use for training
    file = random.sample(all_files, k=1)[0]
    print(str(file))
    img = mpimg.imread(file)
    # divide image into tiles
    pictureSlices, coordinatesSlices = ImgCalc.image_slicer(img, 3, 3)
    # randomly select a tile per image
    idx = random.randint(0,8)
    tile_sel = pictureSlices[idx]
    coords_tile_sel = coordinatesSlices.loc[:idx]
    # classify tile for LC
    svm_classification = ClassificationFunctions.pred_lc(img=tile_sel, path_model=workdir / 'SVM_Output' / picture_type / 'svm_clf_lc_probs.pkl')
    classification_value = ClassificationFunctions.proc_lc_lab(svm_classification, th=0.25)  # class label
    # count images per class, to ensure equal number of training images per class
    if classification_value == 1:
        count1 = count1 + 1
    elif classification_value == 2:
        count2 = count2 + 1
    else:
        count3 = count3 + 1
    print(count1, '/', count2, '/', count3)

    # when number of images for all classes is reached, end process
    if count1 < n_img or count2 < n_img or count3 < n_img:
        continue_proc = True
    else:
        print('stopping')
        continue_proc = False

    # else continue collecting training data for the required classes
    if count1 >= n_img and classification_value == 1:
        print('skipping')
        continue
    elif count2 >= n_img and classification_value == 2:
        print('skipping')
        continue
    elif count3 >= n_img and classification_value == 3:
        print('skipping')
        continue

    # GET TRAINING POSITIONS
    # handle closing of GUI when inadequate images are proposed for training
    try:
        coords = []
        coords = ImageFunctions.capture_training_positions_GUI(tile_sel, training_coordinates=coords)
        # adjust counter if selected image is not used
        if not coords:
            if classification_value == 1:
                count1 = count1 - 1
            elif classification_value == 2:
                count2 = count2 - 1
            else:
                count3 = count3 - 1
            # need to specifically raise error
            raise ValueError('image not used, loading next')
    except:
        print('image not used, loading next')
        continue

    for cs in coords:
        cs.update({"lab": classification_value})
        cs.update({"img": file})
    train = pd.DataFrame(coords)

    # demosaic and get descriptors
    color_spaces, descriptors, descriptor_names = ImgCalc.get_colorspaces_8bit(tile_sel)
    _, a_RGB_16bitf, a_HSV_16bitf, a_Lab_16bitf, a_ExG, a_ExR = color_spaces

    file_basename = os.path.basename(file)
    file_trcname = file_basename[0:-4]

    # save used image tile
    path = f'{workdir}/RF_output/{picture_type}/training_imgs/{classification_value}'
    img_name = file.replace('\\', '_')[3:]
    if not Path(path).exists():
        Path(path).mkdir(parents=True, exist_ok=True)
    mpimg.imsave(f'{path}/{file_basename}', tile_sel)

    # save training position
    path = f'{workdir}/RF_output/{picture_type}/training_coords/{classification_value}'
    if not Path(path).exists():
        Path(path).mkdir(parents=True, exist_ok=True)
    train.to_csv(f'{path}/{file_trcname}.csv', index=False)

    # get training data
    X, y = ImageFunctions.extract_training_from_coordinates(descriptors, coords)
    df_trainings = pd.DataFrame(X, columns=descriptor_names)
    df_trainings['response'] = y

    # save training data
    path = f'{workdir}/RF_output/{picture_type}/training_data/{classification_value}'
    if not Path(path).exists():
        Path(path).mkdir(parents=True, exist_ok=True)
    df_trainings.to_csv(f'{path}/{file_trcname}.csv',
                        columns=descriptor_names.append('response'), index=False)

# ======================================================================================================================
# 2. TRAIN RF CLF
# ======================================================================================================================

# specify model hyper-parameters
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

# get training data for all lc classes and train rf for each class
classes = [1, 2, 3]
for cl in classes:
    path = workdir / 'RF_output' / picture_type / 'training_data' / str(cl)
    files = glob.glob(f'{path}/*.csv')
    xs = []
    ys = []
    for file in files:
        data = pd.read_csv(file)
        pred = np.asarray(data)[:,0:14]
        resp = np.asarray(data)[:,14]
        xs.append(pred)
        ys.append(resp)
    X = np.asarray(np.concatenate(xs, axis = 0))
    y = np.asarray(np.concatenate(ys, axis = 0))

    # fit rf
    model = clf.fit(X, y)
    score = model.score(X, y)

    # save model
    path = f'{workdir}/RF_output/{picture_type}/rf_clf'
    if not Path(path).exists():
        Path(path).mkdir(parents=True, exist_ok=True)
    pkl_filename = f'{path}/{cl}_rf_clf.pkl'
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)

# ======================================================================================================================
# 3. VALIDATE ON UNSEEN IMAGES
# ======================================================================================================================

workdir = Path('./').resolve()

img_dir = Path('O:/Hiwi/2020_Herbifly/Images_Farmers')

all_files = glob.glob((f'{img_dir}/**/**/10m/**/*.JPG'))
len(all_files)  # 17036 potential training images

# files_to_process = random.sample(all_files)

# register number of images processed per lc class
count1 = 0
count2 = 0
count3 = 0
n_img = 10 #number of validation images to use per class
continue_proc = True
while(continue_proc):
    # randomly select an image to use for validation
    file = random.sample(all_files, k=1)[0]
    print(str(file))
    img = mpimg.imread(file)
    # classify img for LC
    svm_result = ImgCalc.pred_lc(img = img, path_model=workdir / 'Output' / 'svm_clf_lc_probs.pkl')
    lc_lab = svm_result['lab']  # class label
    prob = svm_result['prob_c1']  # class probability
    # overwrite class label with class 3, if probability is below threshold
    if(prob <= 0.75 and prob >= 0.25):
        lc_lab = 3

    # count images per class, to ensure equal number of training images per class
    if lc_lab == 1:
        count1 = count1 + 1
    elif lc_lab == 2:
        count2 = count2 + 1
    else:
        count3 = count3 + 1
    print(count1, '/', count2, '/', count3)

    # when number of images for all classes is reached, end process
    if count1 < n_img or count2 < n_img or count3 < n_img:
        continue_proc = True
    else:
        print('stopping')
        continue_proc = False

    # else continue collecting training data for the required classes
    if count1 >= n_img and lc_lab == 1:
        print('skipping')
        continue
    elif count2 >= n_img and lc_lab == 2:
        print('skipping')
        continue
    elif count3 >= n_img and lc_lab == 3:
        print('skipping')
        continue

    # demosaic and get descriptors
    color_spaces, descriptors, descriptor_names = ImageFunctions.demosaic_8bit_image(img)

    # Use the prediction method of the random forest classifier to predict the image
    # The predict() method needs a 2-dimensional array (first dimension: samples, second dimension: features)
    # We therefore need to flatten the array from 3 to 2 dimensions:
    path = f'{workdir}/test_output/rf_clf/{lc_lab}/rf_clf.pkl'
    with open(path, 'rb') as model:
        rf_clf = pickle.load(model)

    descriptors_flatten = descriptors.reshape(-1, descriptors.shape[-1])

    # Predict (ATTENTION: this step may take a while, please wait until the process is done)
    a_segmented_flatten = rf_clf.predict(descriptors_flatten)

    file_basename = os.path.basename(file)
    file_trcname = file_basename[0:-4]

    # Reshape result to 3-dimensional array and convert to uint8 (0: soil, 1: plant)
    a_segmented = np.uint8(np.round(a_segmented_flatten.reshape((descriptors.shape[0], descriptors.shape[1]))))
    # Save as image
    path = f'{workdir}/test_output/validation_imgs/{lc_lab}/{file_trcname}'
    imageio.imwrite(path + '.tiff', a_segmented * 255)

    fig, ax = plt.subplots(1, 2, figsize=(10, 10), sharex=True, sharey=True)
    ax[0].imshow(img)
    ax[0].set_title("resized")
    ax[1].imshow(a_segmented)
    ax[1].set_title("final")
    plt.tight_layout()
    plt.show()

    _, a_RGB_16bitf, a_HSV_16bitf, a_Lab_16bitf, a_ExG, a_ExR = color_spaces
