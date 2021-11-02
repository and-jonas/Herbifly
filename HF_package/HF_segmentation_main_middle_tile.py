# ======================================================================================================================
# Author: Jonas Anderegg, Flavian Tschurr
# Project: Herbifly
# Script use: HF segmentation of single images by classification, segmentation, referencing to world coordiantes etc.
# Date: 05.05.2020
# Random Forest etc. from Lukas Roth --> https://gitlab.ethz.ch/crop_phenotyping/crop_phenotyping_course
# Last modified: 2021-11-02
# ======================================================================================================================

# imports
import numpy as np
import geojson
from pathlib import Path
import imageio
import pandas as pd
import matplotlib.image as mpimg
import os
import glob
import cv2
import Metashape
import pickle
from termcolor import colored

# the following lines are to change de default separator within paths
# (if some strange behaviour in built path occurs, check here!)
native_os_path_join = os.path.join


def modified_join(*args, **kwargs):
    return native_os_path_join(*args, **kwargs).replace('\\', '/')


os.path.join = modified_join

import HF_package.ImageFunctions as ImageFunctions
import HF_package.utils as utils
import HF_package.ImageCalculations as ImgCalc
import HF_package.AgisoftFunctions as AgisoftFunctions
import HF_package.HandheldFunctions as HandheldFunctions
import HF_package.ClfFunctions as ClfFunctions
import HF_package.LcClfUtils as LcClfUtils


# ======================================================================================================================
# create a class
# ======================================================================================================================

class SegmentationCalculator:

    def __init__(self, workdir, picture_type, picture_roi, pic_format, features, farmers, gridSize, agisoft_path):
        self.workdir = workdir
        self.picture_type = picture_type
        self.picture_roi = picture_roi
        self.pic_format = pic_format
        self.features = features
        self.farmers = farmers
        self.gridSize = gridSize
        self.agisoft_path = agisoft_path

    def get_rowdet_foldername(self):
        """
        Get names of the row mask and the index maps, depending on the picture type
        :return:
        """
        if self.picture_type == "Handheld":
            rowmask_name = "rowmask_py5fullresTGI_tile1x1"
            idxmap_name = "idx_map_combine_py5fullresTGI_tile1x1"
        elif self.picture_type == "10m" or self.picture_type == "50m":
            rowmask_name = "rowmask_py0TGI_tile3x3"
            idxmap_name = "idx_map_combine_py0TGI_tile3x3"
        else:
            print("Don't know where to get row masks and index maps!")
        return rowmask_name, idxmap_name

    def image_segmentation(self, image, path_myDate, path_trainings, path_current_segmentation, pic_name,
                           path_trained_classification_model):
        """
        Segment images into vegetation and background.
        Images are first classified according to the light contrast.
        Handheld images and images from 10m UAV flights are segmented fully. Only the central tile is processed for
        50m UAV images.
        Resulting masks are saved. Binary masks are returned.
        :param image: Image to segment.
        :param path_myDate: path (character string)
        :param path_trainings: path (character string)
        :param path_current_segmentation: path( character string)
        :param pic_name: path (character string)
        :param path_trained_classification_model: path (character string)
        :return: Binary segmentation mask
        """
        path_currentImage = os.path.join(path_myDate, image)
        pictureCurrent_all = mpimg.imread(str(path_currentImage))

        pictureSlices, coordinatesSlices = ImgCalc.image_slicer(pictureCurrent_all, 3, 3)
        # get coordinates of middle tile for later use --> check if a grid cell is within the middle tile
        coords_middle_tile = np.array(coordinatesSlices[4:5])
        # iterate over the slices of pictures and segment them --> afterwards stitch them together again
        segmentationPics = []
        for pic in range(len(pictureSlices)):
            # name the slices
            pictureCurrent = pictureSlices[pic]
            # if handheld, run rf on full image
            if self.picture_type == "Handheld" or (self.picture_type == "10m" and self.picture_roi == "fullsize"):
                svm_classification = LcClfUtils.pred_lc(pictureCurrent,
                                                        path_trained_classification_model)
                classification_value = LcClfUtils.proc_lc_lab(svm_classification, th=0.25)

                # get color spaces and descriptors
                color_spaces, descriptors, descriptor_names = ImgCalc.get_colorspaces_8bit(pictureCurrent)
                _, a_RGB_16bitf, a_HSV_16bitf, a_Lab_16bitf, a_ExG, a_ExR = color_spaces

                # ======================================================================================================
                # segmentation
                # ======================================================================================================

                # load the pre-trained random forest
                model_name = "{class_value}_rf_clf.pkl".format(class_value=classification_value)
                path_trained_model = os.path.join(path_trainings, model_name)

                with open(path_trained_model, 'rb') as file:
                    clf = pickle.load(file)

                descriptors_flatten = descriptors.reshape(-1, descriptors.shape[-1])  # flatten
                a_segmented_flatten = clf.predict(descriptors_flatten)  # predict
                a_segmented = np.uint8(
                    np.round(a_segmented_flatten.reshape((descriptors.shape[0], descriptors.shape[1]))))

                # For output
                segmentationPics.append(a_segmented)
            else:
                # number 4 is the middle tile
                if pic == 4:
                    # classification routine
                    svm_classification = LcClfUtils.pred_lc(pictureCurrent,
                                                            path_trained_classification_model)
                    classification_value = LcClfUtils.proc_lc_lab(svm_classification, th=0.25)

                    # get color spaces and descriptors
                    color_spaces, descriptors, descriptor_names = ImgCalc.get_colorspaces_8bit(pictureCurrent)
                    _, a_RGB_16bitf, a_HSV_16bitf, a_Lab_16bitf, a_ExG, a_ExR = color_spaces

                    # ==================================================================================================
                    # segmentation
                    # ==================================================================================================

                    # load the pre-trained random forest
                    model_name = "{class_value}_rf_clf.pkl".format(class_value=classification_value)
                    path_trained_model = os.path.join(path_trainings, model_name)
                    with open(path_trained_model, 'rb') as file:
                        clf = pickle.load(file)

                    descriptors_flatten = descriptors.reshape(-1, descriptors.shape[-1])  # flatten
                    a_segmented_flatten = clf.predict(descriptors_flatten)  # predict
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

    def classify_components(self, mask, image, pic_name,
                            base_output_folder,
                            path_myDate,
                            path_row_mask, path_rowsprof,
                            path_trained_component_classification_model,
                            path_current_segmentation,
                            path_current_json,
                            path_current_prediction):
        """
        Post-processes the raw segmentation mask. Loads image-based features. Extracts features for all retained objects
        of the post-processed mask. Assembles the full feature data set. Applies the previously trained object
        classification model to each object.
        Writes post-processed masks and overlays to specified directories.
        :param mask: Raw binary mask
        :param image: Original image
        :param pic_name: Image name
        :param base_output_folder: path (character string)
        :param path_myDate: path (character string)
        :param path_row_mask: path (character string)
        :param path_rowsprof: path (character string)
        :param path_trained_component_classification_model: path (character string)
        :param path_current_segmentation: path (character string)
        :param path_current_json: path (character string)
        :param path_current_prediction: path (character string)
        :return: Weed mask, Image overlay
        """
        # read original image
        path_currentImage = os.path.join(path_myDate, image)
        pictureCurrent_all = mpimg.imread(str(path_currentImage))
        path_currentTile = "{path_current_segm}/{pic_n}_tile_orig.tif".format(
            path_current_segm=path_current_segmentation, pic_n=pic_name)
        img = mpimg.imread(path_currentTile)

        # ==============================================================================================================
        # Get image-based features
        # ==============================================================================================================

        path_row_mask_current = "{path_row_m}/{pic_n}_rowmask.tif".format(path_row_m=path_row_mask,
                                                                          pic_n=pic_name)
        mean_row_tgi, mean_is_tgi, rows = ClfFunctions.extract_img_features(path_rowmask=path_row_mask_current,
                                                                            picture_type=self.picture_type)

        # ==============================================================================================================
        # Post-process mask
        # ==============================================================================================================

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

        # use reference meters to multiply with the parameters
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
            path_ppmask_erode = "{path_current_segm}/{pic_n}_mask_pp.tif".format(
                path_current_segm=path_current_segmentation, pic_n=pic_name)
            path_ppmask_rec = "{path_current_segm}/{pic_n}_mask_pp_all_rec.tif".format(
                path_current_segm=path_current_segmentation, pic_n=pic_name)
            path_ppimg = "{path_current_segm}/{pic_n}_img_pp.tif".format(
                path_current_segm=path_current_segmentation, pic_n=pic_name)
            if Path(path_ppmask_rec).exists() and Path(path_ppimg).exists() and Path(path_ppmask_erode).exists():
                print('>>Post-processed vegetation Mask already exists. Skipping post-processing.')
                img_cnts = imageio.imread(path_ppimg)
                ppmask_erode = imageio.imread(path_ppmask_erode)
                ppmask = imageio.imread(path_ppmask_rec)
            else:
                # post-process mask
                img_cnts, ppmask_erode, ppmask = ClfFunctions.post_process_hh_mask(img=img,
                                                                                   mask=mask,
                                                                                   min_size=250)
                # save output
                imageio.imwrite(path_ppmask_erode, ppmask_erode)
                imageio.imwrite(path_ppimg, img_cnts)
                imageio.imwrite(path_ppmask_rec, ppmask)
        else:
            ppmask = ClfFunctions.post_process_mask(mask,
                                                    kernel_morph=kernel_morph,
                                                    kernel_closing_blur=kernel_closing,
                                                    max_weed_size=max_weed_size)
            # Colors not needed here, wheat removed directly
            ppmask = np.where(ppmask == 125, 0, ppmask)

        # ==============================================================================================================
        # Get object-based features
        # ==============================================================================================================

        print('>>Extracting features...')

        path_pred_current = "{path_current_pred}/{pic_n}_predictordata.csv".format(
            path_current_pred=path_current_prediction, pic_n=pic_name)

        # path to row profiles
        path_rowsprof_current = "{path_rowsprof}/{pic_n}_idx_combine.png".format(path_rowsprof=path_rowsprof,
                                                                                 pic_n=pic_name)
        # get feature data
        if Path(path_pred_current).exists():
            print('>>Feature data already exists. Skipping feature extraction.')
            X = pd.io.parsers.read_csv(path_pred_current)
        else:
            if self.picture_type == "Handheld":
                rowsprof = mpimg.imread(path_rowsprof_current)
                orig_mask_filtered = cv2.medianBlur(mask, 5)
            else:
                rowsprof = mpimg.imread(path_rowsprof_current)[1216:2432, 1824:3648]
                orig_mask_filtered = np.where(ppmask == 0, ppmask, 255)
            # extract predictor data from images
            comps, X = ClfFunctions.extract_obj_features(img=img,
                                                         picture_type=self.picture_type,
                                                         orig_mask_filtered=orig_mask_filtered,
                                                         pp_mask=ppmask,
                                                         rows=rows,
                                                         idx_map=rowsprof,
                                                         reconstruct=True,
                                                         training_coords=None)

        # ==============================================================================================================
        # Assemble feature data
        # ==============================================================================================================

        # to ensure equal column order in training data and data for predictions
        path_training_data = f'{base_output_folder}/test_output/training_data/{self.features}/training_data_bal.csv'
        template = pd.io.parsers.read_csv(path_training_data)
        name_order = list(template.columns)[2:]
        X = pd.DataFrame(X)
        # add image-based features
        X['mean_row_tgi'] = mean_row_tgi
        X['mean_is_tgi'] = mean_is_tgi
        # reorder variables in df
        X = X[name_order]
        X.to_csv(path_pred_current, index=False)

        # ==============================================================================================================
        # CLASSIFY OBJECTS AND CREATE WEED MASKS
        # ==============================================================================================================

        path_predicted_image = "{path_current_j}/{pic_n}_predicted_image.tif".format(
            path_current_j=path_current_prediction,
            pic_n=pic_name
        )
        path_predicted_mask = "{path_current_j}/{pic_n}_predicted_mask.tif".format(
            path_current_j=path_current_prediction,
            pic_n=pic_name
        )
        if not Path(path_predicted_mask).exists():
            img_clf, mask_clf = ClfFunctions.predict_obj_class(
                img=img,
                ppmask=ppmask,
                pic_name=pic_name,
                X=X,
                path_current_prediction=path_current_prediction,
                path_trained_component_classification_model=path_trained_component_classification_model,
                size_threshold=22500
            )
        else:
            print('>>Predicted images already exist. Skipping prediction.')
            img_clf = imageio.imread(path_predicted_image)
            mask_clf = imageio.imread(path_predicted_mask)

        return img_clf, mask_clf

    def process_drone_mask(self, chunk, cornersDF, mask, pic_name, path_output_date_csv):
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

    def process_handheld_mask(self, path_current_img, pic_name, path_current_json,
                              contour_mask, path_output_date_csv):
        """
        Extracts the percentage of the roi covered with weeds and writes to csv
        :param path_current_img: path (character string)
        :param pic_name: image name
        :param path_current_json: path (character string)
        :param contour_mask: the weed mask obtained from "classify_components"
        :param path_output_date_csv: path (character string)
        """
        # check whether geojson file exists
        current_image = imageio.imread(path_current_img)
        if not Path("{path_current_j}/{pic_n}.geojson".format(path_current_j=path_current_json,
                                                              pic_n=pic_name)).exists():
            corners = ImageFunctions.capture_plot_shape_GUI(current_image)
            HandheldFunctions.write_geojson_polygon_mask_handheld(corners=corners, image_name=pic_name,
                                                                  path_folder=path_current_json)
        with open("{path_current_j}/{pic_n}.geojson".format(path_current_j=path_current_json, pic_n=pic_name),
                  'r') as infile:
            polygon_mask = geojson.load(infile)
        coverage = HandheldFunctions.handheld_coverage_calculator(polygon_mask, contour_mask)
        # coverage = AgisoftFunctions.grid_coverage_calculator(polygon_mask,contour_mask)
        df_coverage = pd.DataFrame(coverage)  # make data frame
        output_namer = "{path_output_d_csv}/{pic_n}_coverage.csv".format(
            path_output_d_csv=path_output_date_csv, pic_n=pic_name)
        df_coverage.to_csv(output_namer, index=False, header=False)

    def iterate_farmers(self):
        """
        Wrapper. Processes all images for all farmers and all measurement dates. Writes predicted weed coverage to csv.
        """
        for farmer in self.farmers:
            farmer_region = utils.get_farmer_region(farmer)
            path_myfarm = os.path.join(self.workdir, farmer_region, farmer, self.picture_type)
            picture_output = "{picture_t}_output".format(picture_t=self.picture_type)
            base_output_folder = os.path.join(self.workdir, "Output", picture_output)
            base_output_folder_farmer = os.path.join(self.workdir, "Output", picture_output, farmer)
            path_trainings = os.path.join(self.workdir, f'Meta/trained_rf/{self.picture_type}/')
            path_previews = os.path.join(base_output_folder_farmer, 'previews')
            path_segmentation = os.path.join(base_output_folder_farmer, 'segmentation')
            path_myproject = os.path.join(self.agisoft_path, farmer_region, farmer, self.picture_type)
            # pre-trained model for classification of pixels
            path_trained_classification_model = os.path.join(
                self.workdir,
                f'Meta/light_contrast/{self.picture_type}/models/lc_svm.pkl'
            )
            # pre-trained model for classification of vegetation objects
            path_trained_component_classification_model = os.path.join(
                self.workdir,
                f'Meta/classification_model/{self.picture_type}/clf_comps_{self.features}_rf.pkl'
            )
            for path in (path_trainings, path_previews, path_segmentation, base_output_folder_farmer):
                Path(path).mkdir(parents=True, exist_ok=True)

            dates = os.listdir(path_myfarm)

            for date in dates:

                if utils._check_date_name(date):

                    # ==================================================================================================
                    # generate directories
                    # ==================================================================================================

                    path_myDate = os.path.join(path_myfarm, date)
                    path_current_segmentation = os.path.join(path_segmentation, date, self.picture_roi)
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

                    # ==================================================================================================
                    # select images and load required files
                    # ==================================================================================================

                    if self.picture_type == "Handheld":
                        # load all available images
                        images = [os.path.basename(x) for x in glob.glob(f'{path_myDate}/*.JPG')]
                    elif self.picture_type == "10m" or "30m" or "50m":
                        # not all images used in Agisoft --> iterate over used ones
                        # Alternatively, iterate over images containing a frame
                        path_grid_corners = os.path.join(self.workdir, "Meta", farmer, self.picture_type, "corners")
                        # Load CornersDF (does not exist where Image Alignment failed)
                        try:
                            cornersDF = pd.read_csv(
                                "{path_grid_c}/{farm}_{dat}_{picture_t}_{roi}_CameraCornerCoordinates.csv".format(
                                    path_grid_c=path_grid_corners, farm=farmer, dat=date, picture_t=self.picture_type,
                                    roi=self.picture_roi))
                        except FileNotFoundError:
                            print("CornersDF does not exist. Skipping.")
                            continue
                        # iterate over images used in Agisoft project
                        images_name = cornersDF.camera
                        # iterate over images containing frames
                        images_name = utils.filter_images_frames(path_current_json=path_current_json,
                                                                 cornersDF=cornersDF)

                        images = []
                        for image in images_name:
                            images.append("{img}{pic_f}".format(img=image, pic_f=self.pic_format))
                        print(f'iterating over {len(images)} Images')

                        # load Agisoft project
                        project = "HF_{farm}_{dat}_{picture_t}.psx".format(farm=farmer, dat=date,
                                                                           picture_t=self.picture_type)
                        doc = Metashape.Document()
                        path_project = os.path.join(path_myproject, date, project)
                        doc.open(path_project)
                        chunk = doc.chunk
                        csv_names = []

                    # ==================================================================================================
                    # process images
                    # ==================================================================================================

                    # counter to be able to load the correct part of the Agisoft project later on
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

                            # ==========================================================================================
                            # Image Segmentation
                            # ==========================================================================================

                            if not path_current_mask.exists():
                                print('>>Segmenting Image...')
                                mask = self.image_segmentation(image,
                                                               path_myDate, path_trainings,
                                                               path_current_segmentation,
                                                               pic_name,
                                                               path_trained_classification_model)
                            else:
                                print('>>Vegetation Mask already exists. Skipping segmentation.')
                                mask = mpimg.imread(path_current_mask)

                            # ==========================================================================================
                            # Object Classification
                            # ==========================================================================================

                            # path_current_weed_mask = Path(
                            #     os.path.join(f'{path_current_prediction}/weed_mask_{pic_name}.tiff'))
                            path_current_weed_mask = Path(
                                os.path.join(f'{path_current_prediction}/{pic_name}_predicted_mask_TEMP.tif'))
                            if not path_current_weed_mask.exists():
                                img_clf, mask_clf = self.classify_components(
                                    mask, image, pic_name,
                                    base_output_folder,
                                    path_myDate,
                                    path_row_mask, path_rowsprof,
                                    path_trained_component_classification_model,
                                    path_current_segmentation,
                                    path_current_json,
                                    path_current_prediction
                                )
                            else:
                                print('>>Weed mask already exists. Skipping classification.')
                                mask_clf = mpimg.imread(path_current_weed_mask)

                            # ==========================================================================================
                            # extract roi weed coverage
                            # ==========================================================================================

                            path_current_image = os.path.join(path_myDate, image)
                            gridIndicator = self.gridSize * 100
                            # path_output_csv = Path(
                            #     os.path.join(f'{path_output_date_csv}/{pic_name}_coverage_{gridIndicator}.csv'))
                            path_output_csv = Path(
                                os.path.join(f'{path_output_date_csv}/{pic_name}_coverage.csv'))
                            if not path_output_csv.exists():
                                if self.picture_type == "Handheld":
                                    # Path(path_current_json).mkdir(parents=True, exist_ok=True)
                                    self.process_handheld_mask(path_current_image,
                                                               pic_name, path_current_json, mask_clf,
                                                               path_output_date_csv)

                                elif self.picture_type == "10m" or "30m" or "50m":
                                    csv_names.append(self.process_drone_mask(chunk, cornersDF, mask_clf,
                                                                             pic_name, path_output_date_csv))
                            else:
                                print('>>Output already exists. Skipping grid filling.')

    # ==================================================================================================================

    def iterate_farmers_postharvest(self, dates):
        for farmer in self.farmers:
            farmer_region = utils.get_farmer_region(farmer)
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

            # pre-trained model for the classification of vegetation components
            # --> training done in "train_object_classifier_uav.py"
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
                                path_current_segmentation, path_current_trainings, path_current_masks,
                                path_output_date_csv,
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
                                            self.process_handheld_mask(pictureCurrent_all, pic_name, path_current_json,
                                                                       contour_mask,
                                                                       path_output_date_csv)

                                        elif self.picture_type == "10m" or "30m" or "50m":
                                            csv_names.append(self.process_drone_mask(chunk, cornersDF, filtered_mask,
                                                                                     pic_name, path_output_date_csv))
                                    else:
                                        print('>>Output already exists. Skipping grid filling.')
                            except:
                                print(colored("Some error occurred. Skipping image", "red"))
                                continue

                except:
                    print(f'Skipping a Campaign: {date}')
                    continue

# ======================================================================================================================

# initiate the class and use the writen functions
def main():
    segmentation_calculator = SegmentationCalculator()
    # segmentation_calculator.iterate_farmers()
    segmentation_calculator.iterate_farmers


if __name__ == '__main__':
    main()

# ======================================================================================================================
