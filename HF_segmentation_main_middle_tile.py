
# ======================================================================================================================
# Author: Flavian Tschurr, Jonas Anderegg
# Project: Herbifly
# Script use: HF segmentation of single images by classification, segmentation, referencing to world coordiantes etc.
# Date: 05.05.2020
# Random Forest etc. from Lukas Roth --> https://gitlab.ethz.ch/crop_phenotyping/crop_phenotyping_course
# Last modified: 20201221
# ======================================================================================================================

# imports
import numpy as np
import matplotlib as mpl
mpl.use('Qt5Agg')
from matplotlib import pyplot as plt
import geojson
from pathlib import Path
import imageio
import pandas as pd
import matplotlib.image as mpimg
import os

# the following lines are to change de default separator within paths
# (if some strange behaviour in built path occurs, check here!)
native_os_path_join = os.path.join
def modified_join(*args, **kwargs):
    return native_os_path_join(*args, **kwargs).replace('\\', '/')

os.path.join = modified_join
from PIL import Image
import cv2
import Metashape
import pickle
import copy
from termcolor import colored
import HF_package.ImageFunctions as ImageFunctions
import HF_package.utils as utils
import HF_package.ImageCalculations as ImgCalc
import HF_package.ObjectFunctions as ObjectFunctions
import HF_package.AgisoftFunctions as AgisoftFunctions
import HF_package.HandheldFunctions as HandheldFunctions
import HF_package.ClfFunctions as ClfFunctions
import glob

#######################################################################################################################
# create a class
########################################################################################################################

class SegmentationCalculator():
    def __init__(self, workdir, picture_type, pic_format, farmers, gridSize, agisoft_path):
        self.workdir = workdir
        self.picture_type = picture_type
        self.pic_format = pic_format
        self.farmers = farmers
        self.gridSize = gridSize
        self.agisoft_path = agisoft_path

    ########################################################################################################################

    def get_rowdet_foldername(self):
        if self.picture_type == "Handheld":
            rowmask_name = "rowmask_py5fullresTGI_tile1x1"
            idxmap_name = "idx_map_combine_py5fullresTGI_tile1x1"
        elif self.picture_type == "10m" or self.picture_type == "50m":
            rowmask_name = "rowmask_py0TGI_tile3x3"
            idxmap_name = "idx_map_combine_py0TGI_tile3x3"
        else:
            print("Don't know where to get row masks and index maps!")
        return rowmask_name, idxmap_name

    # function that wrapes everything what happens with the picture itself
    def calculate_image(self, image, path_myDate, path_trainings, path_current_segmentation, pic_name,
                        path_row_mask, path_trained_classification_model, path_current_json):

        path_currentImage = os.path.join(path_myDate, image)

        pictureCurrent_all = mpimg.imread(str(path_currentImage))
        # name of the pic, without the format

        print(image)

        pictureSlices, coordinatesSlices = ImgCalc.image_slicer(pictureCurrent_all, 3, 3)
        # we get the coordinates from the middle tile for later use --> check if a grid cell is within the middle tile
        # coords_middle_tile = coordinatesSlices[4:5]
        coords_middle_tile = np.array(coordinatesSlices[4:5])
        # iterate over the slices of pictures and segment them --> afterwards stitch them together again
        segmentationPics = []
        for pic in range(len(pictureSlices)):
            # name the slices
            pictureCurrent = pictureSlices[pic]
            # if we have a handheld image, we perform the random forest on all image slices
            if self.picture_type == "Handheld":
                # classification routine
                svm_classification = LcClfUtils.pred_lc(pictureCurrent,
                                                        path_trained_classification_model)
                classification_value = LcClfUtils.proc_lc_lab(svm_classification, th=0.25)

                # get colospaces and descriptors
                color_spaces, descriptors, descriptor_names = ImgCalc.get_colorspaces_8bit(pictureCurrent)
                # color_spaces, descriptors, descriptor_names = ImageFunctions.demosaic_8bit_image(pictureCurrent)
                _, a_RGB_16bitf, a_HSV_16bitf, a_Lab_16bitf, a_ExG, a_ExR = color_spaces

                ############################################################################################################
                ###### segmentation method                                                                            ######
                ############################################################################################################

                # load the pre trained random forest
                # model_name = f'{classification_value}_rf_clf.pkl'
                model_name = "{class_value}_rf_clf.pkl".format(class_value=classification_value)
                path_trained_model = os.path.join(path_trainings, model_name)

                with open(path_trained_model, 'rb') as file:
                    clf = pickle.load(file)

                # prediction is 2D --> we need to flatten our descriptors
                descriptors_flatten = descriptors.reshape(-1, descriptors.shape[-1])

                # prediction --> may take a while
                a_segmented_flatten = clf.predict(descriptors_flatten)

                # Reshape result to 3-dimensional array and convert to uint8 (0: soil, 1: plant)
                a_segmented = np.uint8(
                    np.round(a_segmented_flatten.reshape((descriptors.shape[0], descriptors.shape[1]))))

            # if we have a drone picture we us the middle tile
            else:
                # number 4 is the middle tile
                if pic == 4:
                    # classification routine
                    svm_classification = LcClfUtils.pred_lc(pictureCurrent,
                                                            path_trained_classification_model)
                    classification_value = LcClfUtils.proc_lc_lab(svm_classification, th=0.25)

                    # get colospaces and descriptors
                    color_spaces, descriptors, descriptor_names = ImgCalc.get_colorspaces_8bit(pictureCurrent)
                    # color_spaces, descriptors, descriptor_names = ImageFunctions.demosaic_8bit_image(pictureCurrent)
                    _, a_RGB_16bitf, a_HSV_16bitf, a_Lab_16bitf, a_ExG, a_ExR = color_spaces

                    ############################################################################################################
                    ###### segmentation method                                                                            ######
                    ############################################################################################################

                    # load the pre trained random forest
                    # model_name = f'{classification_value}_rf_clf.pkl'
                    model_name = "{class_value}_rf_clf.pkl".format(class_value=classification_value)
                    path_trained_model = os.path.join(path_trainings, model_name)

                    with open(path_trained_model, 'rb') as file:
                        clf = pickle.load(file)

                    # prediction is 2D --> we need to flatten our descriptors
                    descriptors_flatten = descriptors.reshape(-1, descriptors.shape[-1])

                    # prediction --> may take a while
                    a_segmented_flatten = clf.predict(descriptors_flatten)

                    # Reshape result to 3-dimensional array and convert to uint8 (0: soil, 1: plant)
                    a_segmented = np.uint8(
                        np.round(a_segmented_flatten.reshape((descriptors.shape[0], descriptors.shape[1]))))
                else:
                    a_segmented = np.zeros([pictureCurrent.shape[0], pictureCurrent.shape[1]], dtype=np.uint8)
            # write into a list --> stitch together afterwards to one pic
            segmentationPics.append(a_segmented)

        ### stitch the pictures together again!
        a_segmented = ImgCalc.image_stitcher(segmentationPics, coordinatesSlices)
        # save segmentation tiff
        # imageio.imwrite(f'{path_current_segmentation}/{pic_name}.tif', a_segmented * 255)
        imageio.imwrite(
            "{path_current_segm}/{pic_n}.tif".format(path_current_segm=path_current_segmentation, pic_n=pic_name),
            a_segmented * 255)

        # contours, hierarchy, cntPic = ObjectFunctions.object_detection(a_segmented, pictureCurrent_all,kernel_size=10)

        path_row_mask_current = "{path_row_m}/{pic_n}_rowmask.tif".format(path_row_m=path_row_mask, pic_n=pic_name)
        rowmask_current = Image.open(path_row_mask_current)
        rowmask_current = np.array(rowmask_current)

        ### create multipliers for the input variables in the object detection routine: picture type specific and
        # using handheld picutres, image specific
        if self.picture_type == "Handheld":
            Path(path_current_json).mkdir(parents=True, exist_ok=True)
            # create or open a geojson mask of the handheld frame
            if Path("{path_current_j}/{pic_n}.geojson".format(path_current_j=path_current_json,
                                                              pic_n=pic_name)).exists() == False:
                corners = ImageFunctions.capture_plot_shape_GUI(pictureCurrent_all)
                HandheldFunctions.write_geojson_polygon_mask_handheld(corners=corners, image_name=pic_name,
                                                                      path_folder=path_current_json)
                # self.write_geojson_polygon_mask_handheld(corners=corners,image_name=pic_name, path_folder= path_current_json)
            with open("{path_current_j}/{pic_n}.geojson".format(path_current_j=path_current_json, pic_n=pic_name),
                      'r') as infile:
                polygon_mask = geojson.load(infile)
            reference_meter = HandheldFunctions.frame_length_reader(polygon_mask)
        if self.picture_type == "10m":
            # we just take a constant value for all 10m images --> width of the image divided by 15, as we expect a
            # image of 10x15 meters ground from a height of 10meter
            reference_meter = pictureCurrent_all.shape[1] / 15
        if self.picture_type == "30m":
            # we just take a constant value for all 10m images --> width of the image divided by 50, as we expect a
            # image of 50 meters ground from a height of 50meter
            reference_meter = pictureCurrent_all.shape[0] / 30
        if self.picture_type == "50m":
            # we just take a constant value for all 10m images --> width of the image divided by 50, as we expect a
            # image of 50 meters ground from a height of 50meter
            reference_meter = pictureCurrent_all.shape[0] / 50

        # now using this reference meters to multiply with the parameters
        # These values need to be validated and set once correctly!!!
        # (validation --> set value for 10m drone flight divide by the reference_meter of the drone flight)
        connectivity_threshold_multiplier = 0.021929824561403508  # 8
        kernel_morph_multiplier = [0.010964912280701754, 0.013706140350877192]  # [4, 5]
        kernel_closing_blur_multiplier = [0.005482456140350877, 0.008223684210526315]  # [2, 3]
        max_weed_size_multiplier = 0.20559210526315788  # 75

        # standardise the input values using the reference meter on the ground and hand
        # these to the calssification function
        connectivity_threshold = int(connectivity_threshold_multiplier * reference_meter)
        kernel_morph = [int(kernel_morph_multiplier[0] * reference_meter),
                        int(kernel_morph_multiplier[1] * reference_meter)]
        kernel_closing = [int(kernel_closing_blur_multiplier[0] * reference_meter),
                          int(kernel_closing_blur_multiplier[1] * reference_meter)]
        max_weed_size = max_weed_size_multiplier * reference_meter

        contour_mask, classified_contours = ObjectFunctions.object_detection_calssification(a_segmented,
                                                                                            pictureCurrent_all,
                                                                                            rowmask_current,
                                                                                            connectivity_threshold,
                                                                                            kernel_morph,
                                                                                            kernel_closing,
                                                                                            max_weed_size)
        #
        detected_weed_img = copy.copy(pictureCurrent_all)
        detected_weed_img = cv2.drawContours(detected_weed_img, classified_contours, -1, (255, 0, 0), 2)

        return contour_mask, detected_weed_img, pictureCurrent_all, coords_middle_tile

    def image_segmentation(self, image, path_myDate, path_trainings, path_current_segmentation, pic_name,
                           path_trained_classification_model):

        path_currentImage = os.path.join(path_myDate, image)
        pictureCurrent_all = mpimg.imread(str(path_currentImage))
        # name of the pic, without the format

        pictureSlices, coordinatesSlices = ImgCalc.image_slicer(pictureCurrent_all, 3, 3)
        # we get the coordinates from the middle tile for later use --> check if a grid cell is within the middle tile
        # coords_middle_tile = coordinatesSlices[4:5]
        coords_middle_tile = np.array(coordinatesSlices[4:5])
        # iterate over the slices of pictures and segment them --> afterwards stitch them together again
        segmentationPics = []
        for pic in range(len(pictureSlices)):
            # name the slices
            pictureCurrent = pictureSlices[pic]
            # if we have a handheld image, we perform the random forest on all image slices
            if self.picture_type == "Handheld":
                # classification routine
                svm_classification = LcClfUtils.pred_lc(pictureCurrent,
                                                        path_trained_classification_model)
                classification_value = LcClfUtils.proc_lc_lab(svm_classification, th=0.25)

                # get colospaces and descriptors
                color_spaces, descriptors, descriptor_names = ImgCalc.get_colorspaces_8bit(pictureCurrent)
                # color_spaces, descriptors, descriptor_names = ImageFunctions.demosaic_8bit_image(pictureCurrent)
                _, a_RGB_16bitf, a_HSV_16bitf, a_Lab_16bitf, a_ExG, a_ExR = color_spaces

                ############################################################################################################
                ###### segmentation method                                                                            ######
                ############################################################################################################

                # load the pre trained random forest
                # model_name = f'{classification_value}_rf_clf.pkl'
                model_name = "{class_value}_rf_clf.pkl".format(class_value=classification_value)
                path_trained_model = os.path.join(path_trainings, model_name)

                with open(path_trained_model, 'rb') as file:
                    clf = pickle.load(file)

                # prediction is 2D --> we need to flatten our descriptors
                descriptors_flatten = descriptors.reshape(-1, descriptors.shape[-1])

                # prediction --> may take a while
                a_segmented_flatten = clf.predict(descriptors_flatten)

                # Reshape result to 3-dimensional array and convert to uint8 (0: soil, 1: plant)
                a_segmented = np.uint8(
                    np.round(a_segmented_flatten.reshape((descriptors.shape[0], descriptors.shape[1]))))

                # For output
                segmentationPics.append(a_segmented)
            else:
                ## number 4 is the middle tile
                if pic == 4:
                    # classification routine
                    svm_classification = LcClfUtils.pred_lc(pictureCurrent,
                                                            path_trained_classification_model)
                    classification_value = LcClfUtils.proc_lc_lab(svm_classification, th=0.25)

                    # get colospaces and descriptors
                    color_spaces, descriptors, descriptor_names = ImgCalc.get_colorspaces_8bit(pictureCurrent)
                    # color_spaces, descriptors, descriptor_names = ImageFunctions.demosaic_8bit_image(pictureCurrent)
                    _, a_RGB_16bitf, a_HSV_16bitf, a_Lab_16bitf, a_ExG, a_ExR = color_spaces

                    ############################################################################################################
                    ###### segmentation method                                                                            ######
                    ############################################################################################################

                    # load the pre trained random forest
                    # model_name = f'{classification_value}_rf_clf.pkl'
                    model_name = "{class_value}_rf_clf.pkl".format(class_value=classification_value)
                    path_trained_model = os.path.join(path_trainings, model_name)

                    with open(path_trained_model, 'rb') as file:
                        clf = pickle.load(file)

                    # prediction is 2D --> we need to flatten our descriptors
                    descriptors_flatten = descriptors.reshape(-1, descriptors.shape[-1])

                    # prediction --> may take a while
                    a_segmented_flatten = clf.predict(descriptors_flatten)

                    # Reshape result to 3-dimensional array and convert to uint8 (0: soil, 1: plant)
                    a_segmented = np.uint8(
                        np.round(a_segmented_flatten.reshape((descriptors.shape[0], descriptors.shape[1]))))

        a_segmented = ImgCalc.image_stitcher(segmentationPics, coordinatesSlices)

        imageio.imwrite(
            "{path_current_segm}/{pic_n}.tif".format(path_current_segm=path_current_segmentation,
                                                     pic_n=pic_name), a_segmented * 255)
        if self.picture_type == "Handheld":
            imageio.imwrite(
                "{path_current_segm}/{pic_n}_tile_orig.tif".format(path_current_segm=path_current_segmentation,
                                                                   pic_n=pic_name), pictureCurrent_all)
        else:
            imageio.imwrite(
                "{path_current_segm}/{pic_n}_tile_orig.tif".format(path_current_segm=path_current_segmentation,
                                                                   pic_n=pic_name), pictureCurrent)

        return a_segmented * 255

        #     else:
        #         a_segmented = np.zeros([pictureCurrent.shape[0], pictureCurrent.shape[1]], dtype=np.uint8)
        # # write into a list --> stitch together afterwards to one pic
        # segmentationPics.append(a_segmented)

    def classify_components(self, mask, image, pic_name,
                            path_myDate,
                            path_row_mask, path_rowsprof,
                            path_trained_component_classification_model,
                            path_current_segmentation,
                            path_current_json,
                            path_current_prediction):

        # read original image
        path_currentImage = os.path.join(path_myDate, image)
        pictureCurrent_all = mpimg.imread(str(path_currentImage))
        path_currentTile = "{path_current_segm}/{pic_n}_tile_orig.tif".format(
            path_current_segm=path_current_segmentation, pic_n=pic_name)
        img = mpimg.imread(path_currentTile)

        ################################################################################################################
        # Get image-based features
        ################################################################################################################

        path_row_mask_current = "{path_row_m}/{pic_n}_rowmask.tif".format(path_row_m=path_row_mask,
                                                                          pic_n=pic_name)
        mean_row_tgi, mean_is_tgi, rows = ClfFunctions.extract_img_features(path_rowmask=path_row_mask_current,
                                                                            picture_type=self.picture_type)

        ################################################################################################################
        # Post-process mask
        ################################################################################################################

        # create multipliers for the input variables in the object detection routine: picture type specific and
        # using handheld pictures, image specific
        if self.picture_type == "Handheld":
            Path(path_current_json).mkdir(parents=True, exist_ok=True)
            # create or open a geojson mask of the handheld frame
            if not Path("{path_current_j}/{pic_n}.geojson".format(path_current_j=path_current_json,
                                                                  pic_n=pic_name)).exists():
                corners = ImageFunctions.capture_plot_shape_GUI(pictureCurrent_all)
                HandheldFunctions.write_geojson_polygon_mask_handheld(corners=corners, image_name=pic_name,
                                                                      path_folder=path_current_json)
            with open("{path_current_j}/{pic_n}.geojson".format(path_current_j=path_current_json, pic_n=pic_name),
                      'r') as infile:
                polygon_mask = geojson.load(infile)
            reference_meter = HandheldFunctions.frame_length_reader(polygon_mask)
        if self.picture_type == "10m":
            # we just take a constant value for all 10m images --> width of the image divided by 15, as we expect a
            # image of 10x15 meters ground from a height of 10meter
            reference_meter = pictureCurrent_all.shape[1] / 15
        if self.picture_type == "30m":
            reference_meter = pictureCurrent_all.shape[1] / 30
        if self.picture_type == "50m":
            reference_meter = pictureCurrent_all.shape[0] / 50

        # now using this reference meters to multiply with the parameters
        # These values need to be validated and set once correctly!!!
        # (validation --> set value for 10m drone flight divide by the reference_meter of the drone flight)
        kernel_morph_multiplier = [0.010964912280701754, 0.013706140350877192]  # [4, 5]
        kernel_closing_blur_multiplier = [0.005482456140350877, 0.008223684210526315]  # [2, 3]
        max_weed_size_multiplier = 0.3426535087719299  # 125

        # standardise input values using reference meter and pass clf function
        kernel_morph = [int(kernel_morph_multiplier[0] * reference_meter),
                        int(kernel_morph_multiplier[1] * reference_meter)]
        kernel_closing = [int(kernel_closing_blur_multiplier[0] * reference_meter),
                          int(kernel_closing_blur_multiplier[1] * reference_meter)]
        max_weed_size = int(max_weed_size_multiplier * reference_meter)

        if self.picture_type == "Handheld":
            # check if output already exists
            path_ppmask = "{path_current_segm}/{pic_n}_mask_pp.tif".format(
                path_current_segm=path_current_segmentation, pic_n=pic_name)
            path_ppmask_rec = "{path_current_segm}/{pic_n}_mask_pp_all_rec.tif".format(
                path_current_segm=path_current_segmentation, pic_n=pic_name)
            path_ppimg = "{path_current_segm}/{pic_n}_img_pp.tif".format(
                path_current_segm=path_current_segmentation, pic_n=pic_name)
            if Path(path_ppmask_rec).exists() and Path(path_ppimg).exists() and Path(path_ppmask).exists():
                print('>>Post-processed vegetation Mask already exists. Skipping post-processing.')
                img_cnts = imageio.imread(path_ppimg)
                ppmask_rec = imageio.imread(path_ppmask_rec)
                ppmask = imageio.imread(path_ppmask)
            else:
                # post-process mask
                img_cnts, ppmask, ppmask_rec = ClfFunctions.post_process_hh_mask(img=img,
                                                                                 mask=mask,
                                                                                 min_size=250)
                # save output
                imageio.imwrite(path_ppmask, ppmask)
                imageio.imwrite(path_ppimg, img_cnts)
                imageio.imwrite(path_ppmask_rec, ppmask_rec)
        else:
            mask_pp = ClfFunctions.post_process_mask(mask,
                                                     kernel_morph=kernel_morph,
                                                     kernel_closing_blur=kernel_closing,
                                                     max_weed_size=max_weed_size)
            # Colors not needed here, wheat removed directly
            mask_pp = np.where(mask_pp == 125, 0, mask_pp)

        ################################################################################################################
        # Get object-based features
        ################################################################################################################

        print('>>Extracting features...')
        path_rowsprof_current = "{path_rowsprof}/{pic_n}_idx_combine.png".format(path_rowsprof=path_rowsprof,
                                                                                 pic_n=pic_name)
        if self.picture_type == "Handheld":

            # # get labelled objects
            # # fill holes in objects to avoid nested contours
            # contours_rec, _ = cv2.findContours(ppmask_rec, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # mask_no_holes = np.zeros(ppmask_rec.shape, dtype="uint8")
            # cv2.drawContours(mask_no_holes, contours_rec, -1, 255, -1)
            # _, labels, _, centroids = cv2.connectedComponentsWithStats(mask_no_holes, connectivity=8)

            path_pred_current = "{path_current_pred}/{pic_n}_predictordata.csv".format(
                path_current_pred=path_current_prediction, pic_n=pic_name)
            if Path(path_pred_current).exists():
                print('>>Feature data already exists. Skipping feature extracton.')
                # load predictor data from csv
                dd = pd.io.parsers.read_csv(path_pred_current)
            else:
                # extract predictor data for handheld images
                rowsprof = mpimg.imread(path_rowsprof_current)
                orig_mask_filtered = cv2.medianBlur(mask, 5)
                comps, X = ClfFunctions.extract_obj_features(img=img,
                                                             orig_mask_filtered=orig_mask_filtered,
                                                             pp_mask=ppmask_rec,
                                                             rows=rows,
                                                             idx_map=rowsprof,
                                                             reconstruct=True,
                                                             training_coords=None)
        else:
            # extract predictor data from images
            rowsprof = mpimg.imread(path_rowsprof_current)[1216:2432, 1824:3648]
            comps, X = ClfFunctions.get_object_descriptors_pred(img, mask_pp, rows, rowsprof)

        ################################################################################################################
        # Assemble feature data
        ################################################################################################################

        # if not loaded from disk, prepare predictor data for use in RandomForestClassifier
        if not Path(path_pred_current).exists():
            # to ensure equal column order in training data and data for predictions
            template = pd.io.parsers.read_csv(
                "O:/Hiwi/2020_Herbifly/Images_Farmers/Output/Handheld_output/test_output_handheld/training_data/reconstruct/training_data_template.csv", sep=";")
            name_order = list(template.columns)[2:]
            X = pd.DataFrame(X)
            # drop variables with too many missing values
            dd = X.drop(["a_entr", "b_entr", "ExG_entr", "ExR_entr"], axis=1)
            # convert the character strings to a binary dummy variable
            # ctype_dummies = pd.get_dummies(dd[['ctype']])
            # dd['ctype_eroded'] = ctype_dummies['ctype_eroded']
            # dd = dd.drop(['ctype'], axis=1)
            # add image-bsed predictors
            dd['mean_row_tgi'] = mean_row_tgi
            dd['mean_is_tgi'] = mean_is_tgi
            # reorder variables in df
            dd = dd[name_order]
            dd.to_csv(path_pred_current, index=False)

        ################################################################################################################
        # CLASSIFY OBJECTS AND CREATE WEED MASKS
        ################################################################################################################

        img_clf, mask_clf = ClfFunctions.predict_obj_class(
            img=img,
            ppmask=ppmask_rec,
            pic_name=pic_name,
            X=dd,
            path_current_prediction=path_current_prediction,
            path_trained_component_classification_model=path_trained_component_classification_model,
            size_threshold=22500
        )

        # # feature data needs to be matched with the components via their centroid positions
        # # Not nice, but all other attempts have failed
        # # Probably due to slightly different shape of components (erosion in feature extraction),
        # # Messing up the object ordering
        #
        # # get centroids from feature data
        # cX = dd["ctr_x"].to_numpy(dtype=int)
        # cY = dd["ctr_y"].to_numpy(dtype=int)
        # # get centroids from the post-processed mask objects
        # ctrds_ = list(zip(cY, cX))
        # ctrds = [list(map(int, i)) for i in centroids[1:]]
        #
        # # find the centroids with the smallest distance from list and extract its index
        # # this is then used to match feature data with the object labels from the post-processed mask
        # inds = []
        # for i in range(len(ctrds_)):
        #     index = np.argmin(cdist([ctrds_[i]], ctrds, 'euclidean'))
        #     inds.append(index)
        #
        # # remove rows with missing predictor values
        # ddd = dd.dropna()
        # # remove centroid position - not a predictor
        # ddd = ddd.drop(["ctr_x", "ctr_y"], axis=1)
        # # convert to array
        # ddd = np.asarray(ddd)
        #
        # ################################################################################################################
        # # Create predictions
        # ################################################################################################################
        #
        # # load pre-trained rf classification model
        # print('>>Classifying components...')
        # with open(path_trained_component_classification_model, 'rb') as file:
        #     model = pickle.load(file)
        #
        # # create prediction using pre-trained model
        # lab = model.predict(ddd)
        #
        # # get indices of large components
        # # these are classified as wheat ex-post
        # index = dd.index
        # index_large_comps = index[dd["area"] > 22500]
        #
        # # fill up label vector with NAs where no prediction could be made (missing values)
        # idx = [index for index, row in dd.iterrows() if row.isnull().any()]
        # vec = list(range(len(dd)))
        # vec_labs = np.setdiff1d(vec, idx).tolist()
        # labs = []
        # counter = 0
        # for i in vec:
        #     if i in vec_labs:
        #         l = lab[counter]
        #         counter += 1
        #     else:
        #         l = "wheat"
        #     labs.append(l)
        #
        # for index in index_large_comps:
        #     # print(labs[index])
        #     labs[index] = "wheat"
        #
        # # set all pixel values belonging to wheat to zero
        # filtered_comps = np.where(np.asarray(labs) == 'wheat')[0]
        # mask_filtered = copy.copy(ppmask_rec)
        # # remove all wheat components from mask
        # for i in filtered_comps:
        #     mask_filtered[labels == inds[i]+1] = 0
        # mask_filtered = mask_filtered.astype("uint8")
        #
        # # Draw contour classified as wheat onto original image
        # weeds = copy.copy(img)
        # all_contours, _ = cv2.findContours(ppmask_rec, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # contours, _ = cv2.findContours(mask_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # # weeds = cv2.drawContours(weeds, all_contours, -1, (0, 0, 255), -1)
        # weeds = cv2.drawContours(weeds, contours, -1, (255, 0, 0), -1)
        #
        # # Save image
        # path_pred_img = "{path_current_pred}/{pic_n}_predicted_image.tif".format(
        #     path_current_pred=path_current_prediction, pic_n=pic_name)
        # imageio.imwrite(path_pred_img, weeds)
        # # save mask with weeds marked
        # path_pred_mask = "{path_current_pred}/{pic_n}_predicted_mask.tif".format(
        #     path_current_pred=path_current_prediction, pic_n=pic_name)
        # imageio.imwrite(path_pred_mask, mask_filtered)
        #
        # # Plot result
        # fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
        # # Show RGB and segmentation mask
        # axs[0].imshow(weeds)
        # axs[0].set_title('contour')
        # axs[1].imshow(weeds)
        # axs[1].set_title('mask')
        # plt.show(block=True)

        # # ==============================================================================================================
        #
        # print('>>Classifying components...')
        #
        # _, output, _, _ = cv2.connectedComponentsWithStats(mask_pp,
        #                                                    connectivity=8)
        # path_rowsprof_current = "{path_rowsprof}/{pic_n}_idx_combine.png".format(path_rowsprof=path_rowsprof,
        #                                                                          pic_n=pic_name)
        # rowsprof = mpimg.imread(path_rowsprof_current)[1216:2432, 1824:3648]
        # tgi_ctr = ClfFunctions.get_essential_object_descriptors(mask_pp, rowsprof)
        #
        # # assemble predictor data
        # X = pd.DataFrame(tgi_ctr, columns = ['tgi_ctr'])
        # X['mean_row_tgi'] = mean_row_tgi
        # X['mean_is_tgi'] = mean_is_tgi
        # cols = X.columns.tolist()
        # cols = cols[-0:] + cols[:-0]
        # X = X[cols]
        #
        # # Filter out weed components
        # with open(path_trained_component_classification_model, 'rb') as file:
        #     clf = pickle.load(file)
        # lab = clf.predict(X)
        # proby = clf.predict_proba(X)
        # filtered_comps = np.where(lab == 'wheat')[0]
        # weed_comps = len(np.where(lab == 'weed')[0])
        # mask_filtered = copy.copy(mask_pp)
        # for i in filtered_comps:
        #         mask_filtered[output == i + 1] = 0
        # mask_filtered = mask_filtered.astype("uint8")
        #
        # print(f'>>>Retained {weed_comps} components')
        #
        # # ==============================================================================================================

        # return mask_filtered

    def proceed_drone_mask(self, chunk, cornersDF, mask, pic_name, path_output_date_csv):
        # x and y coordinates of the grid of the current image
        x_coords, y_coords = AgisoftFunctions.image_grid_creator(cornersDF=cornersDF, current_picName=pic_name,
                                                                 gridSize=self.gridSize)

        cam_nr = []
        for camera in chunk.cameras:
            cam_nr.append(camera.label)
        cameraNumber = cam_nr.index(pic_name)
        camera = chunk.cameras[cameraNumber]

        # camera = chunk.cameras[cameraCounter]
        if camera.label == pic_name:
            oneImg_output = AgisoftFunctions.iterate_pic_grid_middle_tile(x_coords, y_coords, chunk, camera=camera,
                                                                          mask=mask,
                                                                          gridSize=self.gridSize
                                                                          # ,
                                                                          # coords_middle_tile=coords_middle_tile
                                                                          )
            cols = ["x_coord", "y_coord", "coverage"]
            oneImg_output_df = pd.DataFrame(oneImg_output, columns=cols)
            gridIndicator = self.gridSize * 100
            # oneImg_output_df.to_csv(f'{path_output_date_csv}/{pic_name}_coverage_{gridIndicator}.csv')
            output_namer = "{path_output_d_csv}/{pic_n}_coverage_{gridIndi}.csv".format(
                path_output_d_csv=path_output_date_csv, pic_n=pic_name, gridIndi=gridIndicator)
            oneImg_output_df.to_csv(output_namer)
            # csv_picName = f'{path_output_date_csv}/{pic_name}_coverage_{gridIndicator}.csv'
            csv_picName = output_namer

        return csv_picName

    def proceed_handheld_mask(self, pictureCurrent_all, pic_name, path_current_json,
                              contour_mask, path_output_date_csv):
        # we check if the geojson file allready exists for this handheld image --> if not, we creat one, otherwise we skip this step
        if Path("{path_current_j}/{pic_n}.geojson".format(path_current_j=path_current_json,
                                                          pic_n=pic_name)).exists() == False:
            corners = ImageFunctions.capture_plot_shape_GUI(pictureCurrent_all)
            HandheldFunctions.write_geojson_polygon_mask_handheld(corners=corners, image_name=pic_name,
                                                                  path_folder=path_current_json)
            # self.write_geojson_polygon_mask_handheld(corners=corners,image_name=pic_name, path_folder= path_current_json)
        with open("{path_current_j}/{pic_n}.geojson".format(path_current_j=path_current_json, pic_n=pic_name),
                  'r') as infile:
            polygon_mask = geojson.load(infile)
        coverageCSV = HandheldFunctions.handheld_coverage_calculator(polygon_mask, contour_mask)
        # coverageCSV = AgisoftFunctions.grid_coverage_calculator(polygon_mask,contour_mask)

        output_namer = "{path_output_d_csv}/{pic_n}_coverage.csv".format(
            path_output_d_csv=path_output_date_csv, pic_n=pic_name)

        coverageCSV.to_csv(output_namer)

    ########################################################################################################################
    # wrapper function that iterates over every field and includes the whole processing
    ########################################################################################################################

    def iterate_farmers(self):
        for farmer in self.farmers:
            farmer_region = utils.get_farmer_region(farmer)
            path_myfarm = os.path.join(self.workdir, farmer_region, farmer, self.picture_type)
            picture_output = "{picture_t}_output".format(picture_t=self.picture_type)
            base_output_folder = os.path.join(self.workdir, "Output", picture_output)
            base_output_folder_farmer = os.path.join(self.workdir, "Output", picture_output, farmer)
            path_output_final_grid = os.path.join(base_output_folder_farmer, "filled_grids")
            path_trainings = os.path.join(self.workdir, f'Meta/trained_rf/{self.picture_type}/')
            path_previews = os.path.join(base_output_folder_farmer, 'previews')
            path_segmentation = os.path.join(base_output_folder_farmer, 'segmentation')
            # path_myproject = os.path.join(self.workdir, "Processed_Campaigns",farmer_region,farmer, self.picture_type)
            path_myproject = os.path.join(self.agisoft_path, farmer_region, farmer, self.picture_type)
            # pre trained model for the classification of pixels
            path_trained_classification_model = os.path.join(
                self.workdir,
                f'Meta/light_contrast/{self.picture_type}/models/lc_svm.pkl'
            )
            # pre-trained model for the classification of vegetation components
            path_trained_component_classification_model = os.path.join(
                self.workdir,
                f'Meta/classification_model/{self.picture_type}/clf_comps_rec_rf.pkl'
            )

            for path in (path_trainings, path_previews, path_segmentation, base_output_folder_farmer):
                Path(path).mkdir(parents=True, exist_ok=True)

            dates = os.listdir(path_myfarm)
            for date in dates:
                # #not all campaigns could be stitched!
                # try:
                # generate folders
                if utils._check_date_name(date):
                    path_myDate = os.path.join(path_myfarm, date)
                    path_current_segmentation = os.path.join(path_segmentation, date)
                    path_current_trainings = os.path.join(path_trainings, date)
                    path_current_previews = os.path.join(path_previews, date)
                    path_current_masks = os.path.join(base_output_folder_farmer, 'mask', date)
                    path_current_prediction = os.path.join(base_output_folder_farmer, "prediction", date)
                    path_row_mask = os.path.join(path_myDate, self.get_rowdet_foldername()[0])
                    path_rowsprof = os.path.join(path_myDate, self.get_rowdet_foldername()[1])
                    path_output_date_csv = os.path.join(base_output_folder_farmer, 'csv', date)
                    path_current_json = os.path.join(self.workdir, "Meta", farmer, self.picture_type, "frames", date)

                    for path in (
                    path_current_segmentation, path_current_trainings, path_current_masks, path_output_date_csv,
                    path_current_previews, path_current_prediction):
                        Path(path).mkdir(parents=True, exist_ok=True)

                    # calculate images
                    if self.picture_type == "Handheld":
                        # we just take every image in the chosen folder in case of handheld picutres
                        images = [os.path.basename(x) for x in glob.glob(f'{path_myDate}/*.JPG')]
                    elif self.picture_type == "10m" or "30m" or "50m":
                        # load the cornersDF --> not all images are used in the agisoft projects --> iteration just over the used ones
                        path_grid_corners = os.path.join(self.workdir, "Meta", farmer, self.picture_type, "corners")
                        # cornersDF = pd.read_csv(f'{path_grid_corners}/{farmer}_{date}_{self.picture_type}_CameraCornerCoordinates.csv')
                        cornersDF = pd.read_csv(
                            "{path_grid_c}/{farm}_{dat}_{picture_t}_CameraCornerCoordinates.csv".format(
                                path_grid_c=path_grid_corners, farm=farmer, dat=date, picture_t=self.picture_type))
                        images_name = cornersDF.camera
                        images = []
                        for image in images_name:
                            images.append("{img}{pic_f}".format(img=image, pic_f=self.pic_format))
                        # load a the corresponding agisoft project
                        project = "HF_{farm}_{dat}_{picture_t}.psx".format(farm=farmer, dat=date,
                                                                           picture_t=self.picture_type)
                        doc = Metashape.Document()
                        path_project = os.path.join(path_myproject, date, project)
                        doc.open(path_project)
                        chunk = doc.chunk
                        csv_names = []

                    # counter to be able to load the correct part of the agisoft project later on
                    n_imgs = len(images)
                    i = 0
                    for image in images:
                        # try:
                        # progress
                        i += 1
                        print(f'Processing images: {i}/{n_imgs}')
                        if utils._check_image_name(image, self.pic_format):
                            pic_name = image[0:-len(self.pic_format)]
                            path_current_mask = Path(os.path.join(f'{path_current_segmentation}/{pic_name}.tif'))
                            if not path_current_mask.exists():
                                print('>>Segmenting Image...')
                                mask = self.image_segmentation(image,
                                                               path_myDate, path_trainings,
                                                               path_current_segmentation,
                                                               pic_name, path_row_mask,
                                                               path_trained_classification_model,
                                                               path_current_json)
                            else:
                                print('>>Vegetation Mask already exists. Skipping segmentation.')
                                mask = mpimg.imread(path_current_mask)

                            path_current_weed_mask = Path(
                                os.path.join(f'{path_current_previews}/weed_mask_{pic_name}.tiff'))
                            if not path_current_weed_mask.exists():
                                filtered_mask = self.classify_components(mask, image, pic_name,
                                                                         path_myDate,
                                                                         path_row_mask, path_rowsprof,
                                                                         path_trained_component_classification_model,
                                                                         path_current_segmentation,
                                                                         path_current_json,
                                                                         path_current_prediction)

            #                         imageio.imwrite("{path_current_pre}/weed_mask_{pic_n}.tiff".format(
            #                             path_current_pre=path_current_previews, pic_n=pic_name), filtered_mask)
            #                     else:
            #                         print('>>Weed mask already exists. Skipping classification.')
            #                         filtered_mask = mpimg.imread(path_current_weed_mask)
            #
            #                     gridIndicator = self.gridSize * 100
            #                     path_output_csv = Path(os.path.join(f'{path_output_date_csv}/{pic_name}_coverage_{gridIndicator}.csv'))
            #                     if not path_output_csv.exists():
            #                         if self.picture_type == "Handheld":
            #                             # Path(path_current_json).mkdir(parents=True, exist_ok=True)
            #                             self.proceed_handheld_mask(pictureCurrent_all,pic_name,path_current_json,contour_mask,
            #                                                        path_output_date_csv)
            #
            #                         elif self.picture_type == "10m" or "30m" or "50m":
            #                             csv_names.append(self.proceed_drone_mask(chunk, cornersDF, filtered_mask,
            #                                                                      pic_name, path_output_date_csv))
            #                     else:
            #                         print('>>Output already exists. Skipping grid filling.')
            #             except:
            #                 print(colored("Some error occurred. Skipping image", "red"))
            #                 continue
            #
            # except:
            #     print(f'Skipping a Campaign: {date}')
            #     continue

    def iterate_farmers_postharvest(self, dates):
        for farmer in self.farmers:

            if farmer == "Baumberger2" or farmer == "Baumberger1" or farmer == "Stettler":
                farmer_region = "Bern_Solothurn"
            elif farmer == "Egli" or farmer == "Keller" or farmer == "Bolli":
                farmer_region = "Nordostschweiz"
            elif farmer == "Scheidegger" or farmer == "Miauton" or farmer == "Bonny":
                farmer_region = "Broye"

            path_myfarm = os.path.join(self.workdir, farmer_region, farmer, self.picture_type)
            # picture_output = f'{self.picture_type}_output'
            picture_output = "{picture_t}_output".format(picture_t=self.picture_type)
            base_output_folder = os.path.join(self.workdir, "Output", picture_output)
            base_output_folder_farmer = os.path.join(self.workdir, "Output", picture_output, farmer)
            path_output_final_grid = os.path.join(base_output_folder_farmer, "filled_grids")
            path_trainings = os.path.join(self.workdir, "Meta/trained_rf")
            path_previews = os.path.join(base_output_folder_farmer, 'previews')
            path_segmentation = os.path.join(base_output_folder_farmer, 'segmentation')
            # path_myproject = os.path.join(self.workdir, "Processed_Campaigns",farmer_region,farmer, self.picture_type)
            path_myproject = os.path.join(self.agisoft_path, farmer_region, farmer, self.picture_type)

            # pre trained model for the classification of pixels --> training is done with the script: train_svm_clf.py
            path_trained_classification_model = os.path.join(self.workdir,
                                                             "Meta/classification_model/svm_clf_lc_probs.pkl")

            # pre-trained model for the classification of vegetation components --> training ist done with the script: get_obj_feats.py
            path_trained_component_classification_model = os.path.join(self.workdir,
                                                                       "Meta/classification_model/clf_comps_rf.pkl")

            for path in (path_trainings, path_previews, path_segmentation, base_output_folder_farmer):
                Path(path).mkdir(parents=True, exist_ok=True)

            # dates = os.listdir(path_myfarm)
            # dates = ['20200805']

            for date in dates:
                # not all campaigns could be stitched!
                try:
                    # generate folders
                    if utils._check_date_name(date):
                        path_myDate = os.path.join(path_myfarm, date)
                        path_current_segmentation = os.path.join(path_segmentation, date)
                        path_current_trainings = os.path.join(path_trainings, date)
                        path_current_previews = os.path.join(path_previews, date)
                        path_current_masks = os.path.join(base_output_folder_farmer, 'mask', date)
                        path_row_mask = os.path.join(path_myDate, "rowmask_py0TGI_tile3x3")
                        path_rowsprof = os.path.join(path_myDate, "idx_map_combine_py0TGI_tile3x3")
                        path_output_date_csv = os.path.join(base_output_folder_farmer, 'csv', date)
                        path_current_json = os.path.join(self.workdir, "Meta", farmer, self.picture_type, "frames",
                                                         date)

                        for path in (
                        path_current_segmentation, path_current_trainings, path_current_masks, path_output_date_csv,
                        path_current_previews):
                            Path(path).mkdir(parents=True, exist_ok=True)

                        # calculate images
                        if self.picture_type == "Handheld":
                            # we just take every image in the chosen folder in case of handheld picutres
                            images = os.listdir(path_myDate)
                        elif self.picture_type == "10m" or "30m" or "50m":
                            # load the cornersDF --> not all images are used in the agisoft projects --> iteration just over the used ones
                            path_grid_corners = os.path.join(self.workdir, "Meta", farmer, self.picture_type, "corners")
                            # cornersDF = pd.read_csv(f'{path_grid_corners}/{farmer}_{date}_{self.picture_type}_CameraCornerCoordinates.csv')
                            cornersDF = pd.read_csv(
                                "{path_grid_c}/{farm}_{dat}_{picture_t}_CameraCornerCoordinates.csv".format(
                                    path_grid_c=path_grid_corners, farm=farmer, dat=date, picture_t=self.picture_type))
                            images_name = cornersDF.camera
                            images = []
                            for image in images_name:
                                images.append("{img}{pic_f}".format(img=image, pic_f=self.pic_format))
                            # load a the corresponding agisoft project
                            project = "HF_{farm}_{dat}_{picture_t}.psx".format(farm=farmer, dat=date,
                                                                               picture_t=self.picture_type)
                            doc = Metashape.Document()
                            path_project = os.path.join(path_myproject, date, project)
                            doc.open(path_project)
                            chunk = doc.chunk
                            csv_names = []

                        # counter to be able to load the correct part of the agisoft project later on

                        # images = images[284:285]

                        n_imgs = len(images)
                        i = 0
                        # images = images[64:65]
                        for image in images:
                            try:
                                # progress
                                i = i + 1
                                print(f'Processing images: {i}/{n_imgs}')
                                if utils._check_image_name(image, self.pic_format):
                                    pic_name = image[0:-len(self.pic_format)]
                                    path_current_mask = Path(
                                        os.path.join(f'{path_current_segmentation}/{pic_name}.tif'))
                                    if not path_current_mask.exists():
                                        print('>>Segmenting Image...')
                                        mask = self.image_segmentation(image,
                                                                       path_myDate, path_trainings,
                                                                       path_current_segmentation,
                                                                       pic_name, path_row_mask,
                                                                       path_trained_classification_model,
                                                                       path_current_json)
                                    else:
                                        print('>>Vegetation Mask already exists. Skipping segmentation.')
                                        mask = mpimg.imread(path_current_mask)

                                    # no component classification required
                                    # all green vegetation is regarded as weeds
                                    filtered_mask = mask

                                    gridIndicator = self.gridSize * 100
                                    path_output_csv = Path(
                                        os.path.join(f'{path_output_date_csv}/{pic_name}_coverage_{gridIndicator}.csv'))
                                    if not path_output_csv.exists():
                                        if self.picture_type == "Handheld":
                                            # Path(path_current_json).mkdir(parents=True, exist_ok=True)
                                            self.proceed_handheld_mask(pictureCurrent_all, pic_name, path_current_json,
                                                                       contour_mask,
                                                                       path_output_date_csv)

                                        elif self.picture_type == "10m" or "30m" or "50m":
                                            csv_names.append(self.proceed_drone_mask(chunk, cornersDF, filtered_mask,
                                                                                     pic_name, path_output_date_csv))
                                    else:
                                        print('>>Output already exists. Skipping grid filling.')
                            except:
                                print(colored("Some error occurred. Skipping image", "red"))
                                continue

                except:
                    print(f'Skipping a Campaign: {date}')
                    continue


########################################################################################################################

# initiate the class and use the writen functions
def main():
    segmentation_calculator = SegmentationCalculator()
    # segmentation_calculator.iterate_farmers()
    segmentation_calculator.iterate_farmers

if __name__ == '__main__':
    main()
