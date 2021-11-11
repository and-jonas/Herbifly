
# ======================================================================================================================
# Author: Jonas Anderegg, jonas.anderegg@usys.ethz.ch
# Project: Herbifly;
# Date: 15.12.2020
# ======================================================================================================================

# imports
import numpy as np
import copy
import statistics
import glob
import re
import json
import pandas as pd
import exifread

import matplotlib as mpl
# mpl.use('Qt5Agg')
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import imageio
from pathlib import Path
import pickle

from scipy import ndimage as ndi
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
from scipy.stats import kurtosis, skew, entropy

from skimage.segmentation import watershed
from skimage import morphology
from skimage import measure
from skimage.feature import greycomatrix, greycoprops
from skimage.filters.rank import entropy as Entropy

import cv2
import math
import sys

from HF_package import ImageFunctions
from HF_package import utils

# reload(utils)

# ======================================================================================================================


# Function to post-process the binary mask
def post_process_mask(mask, kernel_morph=[3, 4], kernel_closing_blur=[2, 3], max_weed_size=150,
                      min_weed_size=13, margin=600):

    print("-post-processing UAV mask...")

    # Remove noise and smooth mask borders
    # Define an elliptic or a rectangular structuring element
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,5))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_morph[0], kernel_morph[1]))

    # Perform morphological closing (i.e. dilation followed by erosion)
    # to smooth contours without simultaneously shrinking components;
    # Side effect: eliminates some of the wholes within components (desired)
    # A rectangular kernel may be preferable to avoid rendering all small object round (?)
    # Elongated shapes might originate from wheat leaves wrongly separated from the row,
    # Single Weed plants are expected to be appear round in shape
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # ## Perform morphological opening (i.e. erosion followed by dilation)
    # ## Does not seem to produce meaningful result
    # opening = cv2.morphologyEx(a_segmented, cv2.MORPH_OPEN, kernel)

    # only slightly blur the obtained mask to get rid of (very) small vegetation patches outside major components
    # probably false positives
    # and to smooth object contours
    # closing_blur = cv2.blur(closing, (2,3))
    closing_blur = cv2.blur(closing, (kernel_closing_blur[0], kernel_closing_blur[1]))

    # get components
    nb_components_cl, output, stats, centroids = cv2.connectedComponentsWithStats(closing_blur, connectivity=8)

    # convert to binary
    output2 = np.where(output > 0, 1, output).astype("uint8")

    # define minimum size of particles
    mask_cleaned = utils.filter_objects_size_remove(output2, min_weed_size, "smaller")

    # define a maximum size of particles to be considered as a distinct weed plant (or small weed patch)
    # if object is larger, it is highly likely that - if weeds - it is connected to wheat objects
    # this situation cannot be addressed in this object-based approach
    mask_cleaned = utils.filter_objects_size(mask_cleaned, max_weed_size, "greater")

    out = utils.omit_borders(mask_cleaned, margin=margin)

    return out


# Function to collect training data for object classification (adapted from Lukas Roth)
def capture_training_positions_GUI_objects(img, mask, training_coordinates=[]):
    """GUI where the user can select image coordinates as training

    :param a_RGB_8bit: RGB image to use as background
    :param a_segmented: Segmented image with 1: plant, 0: soil
    :param training_coordinates: Coordinates of existing training in the form ([[x1, y1],...,[xn, yn]], [[x1, y1],...,[xn, yn]])
    :param subsection: tuple with length=4 defining a region x1 - x2, y1 - y2 to use for plot
    :return: Coordinates of plant and soil pixels in the form ([[x1, y1],...,[xn, yn]], [[x1, y1],...,[xn, yn]])
    """

    # List for plot elements
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)

    # toolbar
    tb = plt.get_current_fig_manager().toolbar

    # Show RGB and segmentation mask
    axs[0].imshow(img)
    axs[0].set_title('Left mouse button to select wheat, right for weed, middle to remove last point. Close window if finished')
    # Show RGB and segmentation mask
    axs[1].imshow(mask)
    axs[1].set_title('Post-processed vegetation mask')

    # Drawing functions for redraw
    def draw_training_points():
        # Redraw all
        fig.canvas.draw()

        # Marked positions
        for plot_id in range(len(axs)):
            if len(training_coordinates) > 0:
                # Get coordinates and assign to wheat or weed
                df_training_coords = pd.DataFrame(training_coordinates)
                splitter = df_training_coords['set'] == 'wheat'
                df_training_wheat = df_training_coords[splitter]
                df_training_weed = df_training_coords[~splitter]
                # Plot crosses for positions
                axs[plot_id].scatter('x', 'y', data=df_training_wheat, marker='+', color='white')
                axs[plot_id].scatter('x', 'y', data=df_training_weed, marker='+', color='red')

    draw_training_points()

    # Event function on click: add or delete training points
    def onclick(event):

        if tb.mode == '':

            # Coordinates of click
            x = event.xdata
            y = event.ydata

            # mask value of click: fixed to 125 here (original shape)
            cnt_type = 125

            # Button 1: Training point for plants added
            if event.button == 1:
                training_coordinates.append({'x': x, 'y': y, 'set': 'wheat', 'cnt_type': cnt_type})
            # Button 3: Training point for soil added
            elif event.button == 3:
                training_coordinates.append({'x': x, 'y': y, 'set': 'weed', 'cnt_type': cnt_type})
            # Button 2: Remove last training point
            elif event.button == 2:
                del training_coordinates[-1]

            print('last entry:', training_coordinates[-1])

            # Remove all points from graph
            for plot_id in range(len(axs)):
                del axs[plot_id].collections[:]

            # Redraw graph
            draw_training_points()


    # Handle mouse click and keyboard events
    fig.canvas.mpl_connect('button_press_event', onclick)

    # Start GUI
    plt.interactive(True)
    plt.show(block=True)

    plt.interactive(False)

    # Return captured coordinates
    return(training_coordinates)


def capture_training_pos_GUI_obj_hh(img, pp_mask_rec, pp_mask_er, training_coordinates=[]):
    """GUI where the user can select image coordinates as training

    :param a_RGB_8bit: RGB image to use as background
    :param a_segmented: Segmented image with 1: plant, 0: soil
    :param training_coordinates: Coordinates of existing training in the form ([[x1, y1],...,[xn, yn]], [[x1, y1],...,[xn, yn]])
    :param subsection: tuple with length=4 defining a region x1 - x2, y1 - y2 to use for plot
    :return: Coordinates of plant and soil pixels in the form ([[x1, y1],...,[xn, yn]], [[x1, y1],...,[xn, yn]])
    """

    # List for plot elements
    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
    # fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)

    # toolbar
    tb = plt.get_current_fig_manager().toolbar

    # Show RGB and segmentation mask
    axs[0].imshow(img)
    axs[0].set_title('Original Image')
    # Show RGB and segmentation mask
    axs[1].imshow(pp_mask_rec)
    axs[1].set_title('PP mask reconstructed')
    axs[2].imshow(pp_mask_er)
    axs[2].set_title('PP mask')

    # Drawing functions for redraw
    def draw_training_points():
        # Redraw all
        fig.canvas.draw()

        # Marked positions
        for plot_id in range(len(axs)):
            if len(training_coordinates) > 0:
                # Get coordinates and assign to wheat or weed
                df_training_coords = pd.DataFrame(training_coordinates)
                splitter = df_training_coords['set'] == 'wheat'
                df_training_wheat = df_training_coords[splitter]
                df_training_weed = df_training_coords[~splitter]
                # Plot crosses for positions
                axs[plot_id].scatter('x', 'y', data=df_training_wheat, marker='+', color='white')
                axs[plot_id].scatter('x', 'y', data=df_training_weed, marker='+', color='red')

    draw_training_points()

    # Event function on click: add or delete training points
    def onclick(event):

        if tb.mode == '':

            # Coordinates of click
            x = event.xdata
            y = event.ydata

            # mask value of click
            cnt_type = pp_mask_er[int(y), int(x)]

            # Button 1: Training point for plants added
            if event.button == 1:
                training_coordinates.append({'x': x, 'y': y, 'set': 'wheat', 'cnt_type': cnt_type})
            # Button 3: Training point for soil added
            elif event.button == 3:
                training_coordinates.append({'x': x, 'y': y, 'set': 'weed', 'cnt_type': cnt_type})
            # Button 2: Remove last training point
            elif event.button == 2:
                del training_coordinates[-1]

            print('last entry:', training_coordinates[-1])

            # Remove all points from graph
            for plot_id in range(len(axs)):
                del axs[plot_id].collections[:]

            # Redraw graph
            draw_training_points()


    # Handle mouse click and keyboard events
    fig.canvas.mpl_connect('button_press_event', onclick)

    # Start GUI
    plt.interactive(True)
    plt.show(block=True)

    plt.interactive(False)

    # Return captured coordinates
    return(training_coordinates)


def extract_obj_features_subset(img, mask, training_coords, rows, idx_map):
    """ detect objects of a given input --> updated function of Jonas
        :param a_segmented: a segmented picture (after random forrest and stitching the tiles together again in our case)
        :param picture: the rgb picture of the segmented
        :param kernel_size: determine the size to smooth the segmented --> get rid of blur and artefacts of bad random forrest calssification
        --> the bigger the size the smoother the image
        :return: list of the contours of all detected object, the hierarchy of this objects, and image and and image with the contours shown on the image
        """

    mask3 = copy.copy(mask)

    if training_coords is not None:

        # get components
        n_comps, output, stats, centroids = cv2.connectedComponentsWithStats(mask3, connectivity=8)
        # remove background
        sizes = stats[1:, -1];
        centroids = centroids[1:, ]
        n_comps = n_comps - 1

        ids = []
        labs = []
        c_coords = []
        for sample in training_coords:
            # Round coordinates
            x_image, y_image = int(round(sample['x'])), int(round(sample['y']))
            c_coord = [x_image, y_image]
            lab = sample['set']

            # get coords, id and class label of used component
            id = output[y_image, x_image]
            ids.append(id)
            labs.append(lab)
            c_coords.append(c_coord)

        # check that no background was marked by mistake
        # (and remove if necessary)
        if any(id == 0 for id in ids):
            idx_drop = [id for id, x in enumerate(ids) if x == 0]
            ids = [i for j, i in enumerate(ids) if j not in idx_drop]
            labs = [i for j, i in enumerate(labs) if j not in idx_drop]
            c_coords = [i for j, i in enumerate(c_coords) if j not in idx_drop]

        print('captured', len(ids), 'training positions. Extracting predictors.')

        # get component descriptors
        # filter components
        for i in range(1, n_comps+1):
            if not i in ids:
                mask3[output == i] = 0

        # check output
        if not(len(labs) == len(ids)):
            sys.exit("something wrong with the number of components!")

        # Plot result
        fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
        # Show RGB and segmentation mask
        axs[0].imshow(mask3)
        axs[0].set_title('img')
        axs[1].imshow(mask3)
        axs[1].set_title('orig_mask')
        plt.show(block=True)

    # ==================================================================================================================

    n_comps, output, stats, centroids = cv2.connectedComponentsWithStats(mask3, connectivity=8)
    cnts, hier = cv2.findContours(mask3, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # remove background
    n_comps = n_comps - 1
    areas = stats[1:, -1];
    centroids = centroids[1:, ]

    # for each component, get area
    area = []
    for i in range(0, n_comps):
        area.append(areas[i])

    # for each detected component contour, get:
    # i)    maximum depth of the convexity defects;
    # ii)   eccentricity;
    # iii)  compactness
    max_depth_convdef = []
    for cont in range(0,len(cnts)):
        hull = cv2.convexHull(cnts[cont], returnPoints=False)
        defects = cv2.convexityDefects(cnts[cont], hull)
        # if there are any convexity defects
        if not defects is None:
            depth_convdef = []
            for defect in defects:
                depth_convdef.append(defect[:, 3][0])
            max_depth_convdef.append(max(depth_convdef))
        # if no convexity defects are detected
        else:
            max_depth_convdef.append(0)

    eccentricity = []
    compactness = []
    for cnt in cnts:
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            # center, axis_length and orientation of ellipse
            (center, axes, orientation) = ellipse
            # length of major and minor axis
            majoraxis_length = max(axes)
            minoraxis_length = min(axes)
            ecc = np.sqrt(1 - (minoraxis_length / majoraxis_length) ** 2)
            # compactness
            A = cv2.contourArea(cnt)
            equi_diameter = np.sqrt(4 * A / np.pi)
            comp = equi_diameter / majoraxis_length
        else:
            ecc = np.NaN
            comp = 0.0
        eccentricity.append(ecc)
        compactness.append(comp)

    # ===================================================================================================================

    color_spaces, descriptors, descriptor_names = ImageFunctions.demosaic_8bit_image(img)
    descriptors = descriptors[:, :, 3:14]

    colorpreds = []
    for i in range(1, n_comps+1):
        pix_idx = np.where(output == i)
        dd = descriptors[pix_idx]
        R, G, B, H, S, V, L, a, b, ExG, ExR = utils.average_preds(dd)
        col = ({'R': R, 'G': G, 'B': B,
                'H': H, 'S': S, 'V': V,
                'L': L, 'a': a, 'b': b,
                'ExG': ExG, 'ExR': ExR})
        colorpreds.append(col)

    # positional information
    rowpoints = np.transpose(np.nonzero(rows))
    min_dist_to_row = []
    tgi_ctr = []
    for i in range(0, len(c_coords)):
        c = c_coords[i]
        dist = np.amin(cdist(np.array([c]), rowpoints, 'euclidean'))
        tgi = idx_map[c[1], c[0]]
        min_dist_to_row.append(dist)
        tgi_ctr.append(tgi)

    # ===================================================================================================================

    # check output
    if not(len(labs) == len(area) == len(max_depth_convdef) == len(eccentricity) == len(compactness) == len(min_dist_to_row) == len(tgi_ctr)):
        sys.exit("incorrect/unequal length of predictor vectors!")

    else:
        if training_coords is not None:
            traindat = []
            for i in range(0, len(labs)):
                XY = ({'area': area[i], 'max_depth_convdef': max_depth_convdef[i],
                       'eccentricity': eccentricity[i], 'compactness': compactness[i],
                       'min_dist_to_row': min_dist_to_row[i], 'tgi_ctr': tgi_ctr[i],
                       'class_label': labs[i]})
                XY = {**XY, **colorpreds[i]}
                traindat.append(XY)
        else:
            traindat = []
            for i in range(0, len(labs)):
                XY = ({'area': area[i], 'max_depth_convdef': max_depth_convdef[i],
                       'eccentricity': eccentricity[i], 'compactness': compactness[i],
                       'min_dist_to_row': min_dist_to_row[i], 'tgi_ctr': tgi_ctr[i]})
                XY = {**XY, **colorpreds[i]}
                traindat.append(XY)
        return traindat


# ======================================================================================================================

# Hand-held


# Function to post-process the binary mask of handheld photos
def post_process_hh_mask(img, mask, min_size):

    # median filter to remove noise without affecting edges
    orig_mask_filtered = cv2.medianBlur(mask, 5)

    print("-post-processing handheld mask...")

    # ==================================================================================================================
    # STEP 0: POST-PROCESS VEGETATION MASK
    # ==================================================================================================================

    print("---removing noise and filling holes...")

    # slight image dilation to favor fusion of disconnected weed leaves
    kernel = np.ones((4, 4), np.uint8)
    dilation0 = morphology.dilation(mask, kernel)

    # remove holes in vegetation
    close_area = morphology.area_closing(dilation0, area_threshold=500)

    # ==================================================================================================================
    # GET VEGETATION PATCHES in a two-step procedure
    # STEP 1: find large objects
    # ==================================================================================================================

    print("---detecting large objects...")

    # erode image to remove wheat leaves and disconnect connected components
    # more aggressively in y-direction than in x-direction,
    # to favour separation of weeds lying close to the wheat rows
    kernel = np.ones((19, 17), np.uint8)
    erosion = morphology.erosion(close_area, selem=kernel)
    # apply median filter to remove small objects without blurring edges
    median_filtered = cv2.medianBlur(erosion, 15)

    # find contours after erosion and median filtering,
    # dilate contours, rather than components, to avoid merging objects again
    res = np.zeros(median_filtered.shape[:2], dtype=np.uint8)

    # find large contours of "original" size
    _, contours, _ = cv2.findContours(median_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours, _ = cv2.findContours(median_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # dilate contours by fitting a thicker contour line, filling the contour, and re-detecting contours
    contours_dilate = []
    for cnt in contours:
        res0 = copy.copy(res)
        cv2.drawContours(res0, cnt, -1, 255, 36)
        cv2.fillPoly(res0, pts=cnt, color=255)
        _, cnt_dil, _ = cv2.findContours(res0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_dilate.append(cnt_dil)

    # large_contours = []
    for cnt in contours_dilate:
        # draw the dilated contours onto image and mask
        cv2.drawContours(img, [cnt][0], -1, (255, 0, 0), 1)
        # # reshape contour data
        # lc = utils.flatten_contour_data(input=cnt, asarray=False)
        # large_contours.append(lc)

    # ==================================================================================================================
    # STEP 2: find small objects
    # Find objects that were removed during aggressive filtering of the binary mask for component separation
    # These are objects in the raw segmentation masks that have
    # no connection to a component obtained from aggressive post-processing
    # ==================================================================================================================

    print("---detecting small objects...")

    # detect all components on the original mask
    _, output, stats, _ = cv2.connectedComponentsWithStats(orig_mask_filtered, connectivity=8)
    sizes = stats[1:, -1];
    # remove small objects (noise)
    idx = (np.where(sizes >= min_size)[0] + 1).tolist()
    out = np.in1d(output, idx).reshape(output.shape)
    mask_cleaned = np.where(out == True, orig_mask_filtered, 0)

    # detect remaining components and find their contours
    ncomps, output2, stats2, _ = cv2.connectedComponentsWithStats(mask_cleaned, connectivity=8)
    _, contours_initial, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # draw large object contours
    mask_cleaned_help = copy.copy(mask_cleaned)
    for cnt in contours:
        # contour is drawn with 5 pixel thickness, to avoid maintaining touching (not intersecting) patches
        cv2.drawContours(mask_cleaned_help, [cnt], -1, 125, 5)

    # remove any component that has an intersection with a large object contour
    # OR that lies within an other object (this can happen!)
    for i in range(1, ncomps):
        pix_idx = np.where(output2 == i)
        dd = mask_cleaned_help[pix_idx]
        vals = median_filtered[pix_idx]
        if any(dd == 125) or any(vals == 255):
            mask_cleaned[pix_idx] = 0
        else:
            continue

    # # perform opening for small objects
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    mask_cleaned_open = cv2.dilate(mask_cleaned, kernel)

    # detect contours of the remaining (SMALL) objects
    _, contours, _ = cv2.findContours(mask_cleaned_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # draw these contours
    for cnt in contours:
        cv2.drawContours(img, [cnt], -1, (0, 0, 255), 1)

    # use a different pixel intensity for small contours (as ctype)
    mask_cleaned_open = np.where(mask_cleaned_open == 255, 125, mask_cleaned_open)

    # combine large and small contours
    cmask = mask_cleaned_open + median_filtered

    # ==================================================================================================================
    # STEP 3: reconstruct eroded objects
    # Creates watershed segments for components
    # The part of the object with pixel overlap with eroded component will be reconstructed within these segments
    # Reconstructed components are combined with original-shape features to return the reconstructed mask
    # ==================================================================================================================

    mask_rec = reconstruct_eroded_objects(pp_mask_all=cmask, orig_mask_filtered=orig_mask_filtered)

    # detect contours of all (small + reconstructed) objects
    _, contours, _ = cv2.findContours(mask_rec, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # draw these contours
    # this image must be used for annotating training data!
    # Training positions MUST be located INSIDE eroded AND reconstructed objects!!
    for cnt in contours:
        cv2.drawContours(img, [cnt], -1, (0, 255, 0), 1)

    # return img, all_contours, contours_dilate, all_cnt_orig, cmask, mask_rec
    return img, cmask, mask_rec


# Mark training positions on image and post-processed vegetation mask
def mark_training_positions(img, eroded_mask, rec_mask, coords):

    img_out = copy.copy(img)
    rec_mask_out = copy.copy(rec_mask)
    eroded_mask_out = copy.copy(eroded_mask)

    for sample in coords:

        # round coordinates to integers
        x_image, y_image = int(round(sample['x'])), int(round(sample['y']))

        # get the label
        lab = sample['set']

        # mark the position according to the class label
        if lab == "weed":
            cv2.circle(img_out, (x_image, y_image), 35, (255, 0, 0), 3)
            cv2.circle(eroded_mask_out, (x_image, y_image), 35, 100, 3)
            cv2.circle(rec_mask_out, (x_image, y_image), 35, 100, 3)
        elif lab == "wheat":
            cv2.circle(img_out, (x_image, y_image), 35, (0, 0, 255), 3)
            cv2.circle(eroded_mask_out, (x_image, y_image), 35, 200, 3)
            cv2.circle(rec_mask_out, (x_image, y_image), 35, 200, 3)

    return img_out, rec_mask_out, eroded_mask_out


# Extract contours of annotated objects, get labels, and contour type
def grab_contours(pp_mask, coords):

    # detect components on post-processed mask
    n_comps, output, stats, centroids = cv2.connectedComponentsWithStats(pp_mask, connectivity=8)

    # extract marked components, get labels, coordinates and contour type
    ids = []
    labs = []
    c_coords = []
    cnt_type = []
    for sample in coords:
        # Round coordinates
        x_image, y_image = int(round(sample['x'])), int(round(sample['y']))
        # get coords, id, class label and contour type of used component
        c_coord = [x_image, y_image]
        lab = sample['set']
        ct = sample['cnt_type']
        id = output[y_image, x_image]
        ids.append(id)
        labs.append(lab)
        c_coords.append(c_coord)
        cnt_type.append(ct)

    # check that no background was marked by mistake
    # (and remove if necessary)
    if any(id == 0 for id in ids):
        idx_drop = [id for id, x in enumerate(ids) if x == 0]
        ids = [i for j, i in enumerate(ids) if j not in idx_drop]
        labs = [i for j, i in enumerate(labs) if j not in idx_drop]
        cnt_type = [i for j, i in enumerate(cnt_type) if j not in idx_drop]
        c_coords = [i for j, i in enumerate(c_coords) if j not in idx_drop]

    # retain only sampled components
    out = np.in1d(output, ids).reshape(output.shape)
    mask_lab_comps = np.where(out == True, pp_mask, 0)

    # get contours of the sampled components
    ccc, _ = cv2.findContours(mask_lab_comps, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # get contour type information and label for the new contour ordering
    cnt_type_reordered = []
    lab_reordered = []
    for i, kont in enumerate(ccc):
        z = 0
        for j, kord in enumerate(c_coords):
            # check whether the selected coordinates are inside or on the contour
            # if there are more than 1 marked position per contour, stop;
            # this indicates an annotation error.
            x = cv2.pointPolygonTest(kont, tuple(kord), True)
            if x >= 0:
                z += 1
                if z > 1:
                    sys.exit("Annotation error - More than 1 training position assigned to contour!")
                cnt_type_r = cnt_type[j]
                lab_r = labs[j]
                cnt_type_reordered.append(cnt_type_r)
                lab_reordered.append(lab_r)

    # reshape contour data
    labelled_contours = []
    for cnt in ccc:
        c = utils.flatten_contour_data(input=[cnt], asarray=False)
        labelled_contours.append(c)

    return ccc, labelled_contours, cnt_type_reordered, lab_reordered


# Dilate contour by a given number of pixels
def dilate_contour(img, contour, ctype, npix):

    # define empty masks to draw contours and hulls onto
    blank = np.zeros(img.shape[:2], dtype=np.uint8)

    # draw thick contours (dilatation)
    if ctype == 255:
        cv2.drawContours(blank, contour, -1, 255, npix)
    elif ctype == 125:
        cv2.drawContours(blank, [contour], -1, 255, npix)
    cv2.fillPoly(blank, pts=[contour], color=255)
    # this is needed for contours lying on the image edge!!
    res0 = cv2.medianBlur(blank, 5)

    # get the dilated contour
    _, cont, _ = cv2.findContours(res0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # median blurring can (rarely) result in disconnected contours
    # only keep the biggest one of these
    try:
        c = max(cont, key=cv2.contourArea)
    # blurring sometimes results in the loss of the contour,
    # catch by using original contour
    except ValueError or ZeroDivisionError:
        _, cont, _ = cv2.findContours(blank, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cont, key=cv2.contourArea)

    # if the convex hull should be returned
    hull = cv2.convexHull(c, False)

    return c, hull


# Get coordinates of centroid position
def get_centroid(contour):
    # get centroid of contour
    M = cv2.moments(contour)
    cv2.contourArea(contour)
    try:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    # if component is only one pixel in either direction (almost never)
    # take the first component pixel as centroid
    except ZeroDivisionError:
        cX = contour[0][0][0]
        cY = contour[0][0][1]
    ctr = [cY, cX]
    return ctr


# Get contour coordinates
def get_cont_coordinates(contour, img):

    blank = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(blank, [contour], -1, 125, 1)
    # extract coordinates
    coords_cont = np.where(blank == 125)
    return coords_cont


def scan_format_contour(coords, polar_coords, orig_mask_filtered):

    # sort according to angle
    cxx = np.argsort(polar_coords[1])
    ang = polar_coords[1][cxx]
    pos = polar_coords[0][cxx]
    # get contour pixel values and reorder
    cpix_vals = np.bitwise_not(orig_mask_filtered)[coords]
    vals = cpix_vals[cxx]
    # interpolate to specific angles
    newang = np.arange(start=-math.pi, stop=math.pi, step=math.pi / 1080)
    vv = np.interp(x=newang, xp=ang, fp=vals)

    return vv


def summarise_scan(result, type, ctype, regrown):

    v = np.column_stack(result)  # stack columns

    # extract a summary from profiles
    # condition required, because regrown components are separated by a gap
    # (which does not represent true soil pixels)
    if ctype == "original":
        v_ = v.max(axis=1).astype(int)  # get the maximum per row
    elif ctype == 'eroded':
        if not regrown:
            v_ = v.max(axis=1).astype(int)  # get the maximum per row
        elif regrown:
            v__ = v.mean(axis=1).astype(int)
            v_ = np.where(v__ >= 3/12*255, 255, 0)
        else:
            print("Need to specify the type of patch to analyze; either regrown or original eroded")
    else:
        print("Need to specify the type of contour to analyze; either eroded or original")

    # find longest consecutive soil segment
    zeros = np.where(v_ == 0)[0]  # find index of plant pixel
    tot_length_cont = len(v_)  # total length of the contour

    # if no plant pixels on contour, the angle is 360°, vegetation fraction is 0,
    # and proportion of the largest soil segment is 1
    # and there are 0 disconnected segments (1 connected)
    if zeros.size == 0:
        angle = 360
        vshare = 0
        prop_longest = 1
        n_segments = 0
    else:
        lengths = np.ediff1d(zeros)  # get vector distance between plant pixels to obtain length of soil segment

        # if the soil segment overlaps with the start/end of the contour
        if not zeros[0] == 0:
            length = tot_length_cont - zeros[-1] + zeros[0]
            lengths = lengths.tolist()
            lengths.append(length)
            lengths = np.asarray(lengths)

        soil_segments = len(np.where(lengths > 1)[0])  # get the number of disconnected soil segments
        max_length = max(lengths)  # get the longest soil segment
        # proportion of the longest soil segment
        prop_longest = max_length / tot_length_cont
        # get angle of the longest soil segment
        index = np.where(lengths == max_length)[0][0]
        lower = zeros[index]  # index of the first soil pixel
        newang = np.arange(start=-math.pi, stop=math.pi, step=math.pi / 1080)

        # if the soil segments overlaps with the start/end of the contour
        try:
            upper = zeros[index + 1]  # index of the last soil pixel
            angs = newang[lower:upper]
        except:
            angs = newang[lower:len(newang)].tolist() + newang[0:zeros[0]].tolist()
            angs = np.asarray(angs)

        # get the distance as an angle
        angs = 180 * angs / math.pi
        angs = angs.astype(int)
        angle = np.max(angs) - np.min(angs)
        # get ratio veg/soil
        vshare = len(zeros) / tot_length_cont
        # get number of plant segments segments
        nonzeros = np.where(v_ != 0)[0]
        lengths = np.ediff1d(nonzeros)
        plant_segments = len(np.where(lengths > 1)[0])
        # get n_segments as the maximum of soil and plant segments (depends on where the contour starts!)
        n_segments = max(soil_segments, plant_segments)

    # define function output
    if type == "hull":
        # return angle
        out = angle
    elif type == "cont":
        # return ratio veg/soil
        out = vshare, n_segments, prop_longest
    return out


def elongation(m):
    x = m['mu20'] + m['mu02']
    y = 4 * m['mu11'] ** 2 + (m['mu20'] - m['mu02']) ** 2
    try:
        el = (x + y ** 0.5) / (x - y ** 0.5)
    except ZeroDivisionError:
        el = np.nan
    return el


def get_object_watershed_labels(pp_mask_all):

    print("---extracting watershed labels...")

    # binarize the post-processed mask
    mask = np.where(pp_mask_all == 125, 255, pp_mask_all)
    # component labelling
    n_comps, output, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # invert the mask
    mask_inv = np.bitwise_not(mask)
    # calculate the euclidian distance transform
    distance = ndi.distance_transform_edt(mask_inv)
    # watershed segmentation, using labelled components as markers
    # this must be done to ensure equal number of watershed segments and connected components!
    labels = watershed(distance, markers=output, watershed_line=True)
    # erosion to thicken watershed boundaries
    kernel = np.ones((2, 2), np.uint8)
    labels_erode = morphology.erosion(labels, kernel)

    # Check
    if not n_comps == len(np.unique(labels)):
        sys.exit('Component mis-match in watershed segmentation!')

    return labels_erode


def reconstruct_eroded_objects(pp_mask_all, orig_mask_filtered):

    watershed_labels = get_object_watershed_labels(pp_mask_all)

    print("---reconstructing eroded vegetation patches...")

    # ==================================================================================================================

    # component labelling
    n_tot, output, _, centroids = cv2.connectedComponentsWithStats(pp_mask_all, connectivity=8)

    # Number of eroded objects in mask
    eroded_obj = np.where(pp_mask_all == 255, pp_mask_all, 0)
    n_erode, _, _, centroids = cv2.connectedComponentsWithStats(eroded_obj, connectivity=8)

    # Number of original objects in mask
    orig_obj = np.where(pp_mask_all == 125, pp_mask_all, 0)
    n_orig, _, _, centroids = cv2.connectedComponentsWithStats(orig_obj, connectivity=8)

    # Check number of components !
    if not n_tot == (n_erode + n_orig-1):
        print('----Component mis-match in watershed segmentation !')

    # ==================================================================================================================

    # binarize mask
    eroded_obj = np.where(pp_mask_all == 255, 1, 0)

    # identify eroded components
    idx = np.unique(eroded_obj * output)

    # Reconstruct vegetation patches inside their watershed segment
    output_mask = np.zeros(pp_mask_all.shape, dtype="uint8")
    for label in np.unique(watershed_labels):
        # skip background
        if label == 0:
            continue
        # skip components with original shape
        elif label not in idx:
            continue
        else:
            # find pixel coordinates of the watershed segment
            coords = np.where(watershed_labels == label)
            # create empty masks
            mask0 = np.zeros(pp_mask_all.shape, dtype="uint8")
            mask1 = copy.copy(mask0)
            # fill watershed segment with pixel values of original and pp_mask
            mask0[coords] = orig_mask_filtered[coords]
            mask1[coords] = pp_mask_all[coords]
            # maintain only the main object of interest that has an intersection with the eroded component
            _, output, _, _ = cv2.connectedComponentsWithStats(mask0, connectivity=8)
            comps_freq = output[np.where(mask1 != 0)].tolist()

            # for large components (can't handle otherwise)
            try:
                digits = int(math.log10(len(comps_freq))) + 1
                if digits >= 4:
                    nth = 10 ** (digits-4)
                    comps_freq_red = comps_freq[0::nth]
                    comps_freq = comps_freq_red
            except ValueError:
                continue
            # try to reconstruct component
            try:
                comp = utils.most_frequent([i for i in comps_freq if i != 0])
                # draw reconstructed patch onto the output mask
                compcoords = np.where(output == comp)
                output_mask[compcoords] = orig_mask_filtered[compcoords]
            # this may fail if the eroded component does not share pixels with the original component
            # in this case, the eroded component must be dilated
            except IndexError:
                # try after object dilation
                try:
                    kernel = np.ones((5, 5), np.uint8)
                    mask1 = morphology.dilation(mask1, kernel)
                    comps_freq = output[np.where(mask1 != 0)].tolist()
                    comp = utils.most_frequent([i for i in comps_freq if i != 0])
                    # draw reconstructed patch onto the output mask
                    compcoords = np.where(output == comp)
                    output_mask[compcoords] = orig_mask_filtered[compcoords]
                except IndexError:
                    # try after further object dilation
                    try:
                        kernel = np.ones((10, 10), np.uint8)
                        mask1 = morphology.dilation(mask1, kernel)
                        comps_freq = output[np.where(mask1 != 0)].tolist()
                        comp = utils.most_frequent([i for i in comps_freq if i != 0])
                        # draw reconstructed patch onto the output mask
                        compcoords = np.where(output == comp)
                        output_mask[compcoords] = orig_mask_filtered[compcoords]
                    # if both levels of dilation do not work, the eroded component is kept
                    except IndexError:
                        print("----maintaining eroded patch.")
                        output_mask[coords] = pp_mask_all[coords]

    # ==================================================================================================================

    # perform checks and reconstruct full mask
    n_erode_fin, _, _, _ = cv2.connectedComponentsWithStats(output_mask, connectivity=8)

    # original patches
    orig_obj = np.where(pp_mask_all == 125, pp_mask_all, 0)
    n_orig, _, _, _ = cv2.connectedComponentsWithStats(orig_obj, connectivity=8)

    # merge masks to create final output with reconstructed eroded objects
    out_mask = output_mask + orig_obj

    # count components
    n_tot_fin, _, _, _ = cv2.connectedComponentsWithStats(out_mask, connectivity=8)

    # Check number of components
    if not n_tot_fin == n_tot:
        print('----Component mis-match in watershed segmentation !')

    return out_mask


def extract_obj_features(img, picture_type,
                         orig_mask_filtered, pp_mask,
                         rows, idx_map,
                         reconstruct, training_coords=None):

    # if no training positions are supplied, features are extracted for all objects
    if training_coords is None:

        # get contours of connected components
        _, ccc, _ = cv2.findContours(pp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours0 = ccc

        # fill contours, then detect components
        checkblank = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(checkblank, ccc, -1, 255, -1)

        # get components
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(checkblank, connectivity=8)

        # component label index must be inverted, because cv2.findContours and cv2.connectedComponentsWithStats
        # start at opposite ends of the image... awesome!
        outputr = n - labels
        outputr = np.where(outputr == n, 0, outputr - 1)

        # check
        if not (n - 1 == len(ccc)):
            sys.exit("something wrong with the number of components!")

    # if coordinates are supplied, the corresponding contours and contour information is extracted
    else:
        # extract labelled objects' contours, corresponding type info and label
        print("---extracting labelled contours...")
        ccc, contours0, cnt_type, labs = grab_contours(pp_mask=pp_mask, coords=training_coords)

    # get predictors for each (labelled) contour
    preds = []
    print("---extracting features...")
    for i, c in enumerate(contours0):

        print(f'----{i+1}/{len(ccc)}')

        # get contour type information
        if training_coords is None:
            contour_type = max(np.unique(pp_mask[np.where(labels == n - 1 - i)]))
        else:
            contour_type = cnt_type[i]

        # get dilatation factors depending on contour type and post-processing procedure
        ctype, dfact = utils.set_contour_dilation_factors(picture_type=picture_type,
                                                          contour_type=contour_type,
                                                          reconstruct=reconstruct)

        # iterate over different levels of dilation
        vector1 = []
        vector2 = []
        for k in range(len(dfact)):
            # dilate contour
            cont, hull = dilate_contour(img=img, contour=ccc[i], ctype=contour_type, npix=dfact[k])
            # get contour centroid
            ctr_hull = get_centroid(contour=hull)
            ctr_cont = get_centroid(contour=cont)
            # get coordinates of contour points and hull points
            c_cont = get_cont_coordinates(contour=cont, img=img)
            c_hull = get_cont_coordinates(contour=hull, img=img)
            # transform from cartesian to polar coordinates
            polar_coords_cont = utils.cart2pol(x=c_cont[1], y=c_cont[0], ctr=ctr_cont)
            polar_coords_hull = utils.cart2pol(x=c_hull[1], y=c_hull[0], ctr=ctr_hull)

            # get pixel values on hull
            v1 = scan_format_contour(coords=c_hull, polar_coords=polar_coords_hull,
                                     orig_mask_filtered=orig_mask_filtered)
            # get pixel values contour
            v2 = scan_format_contour(coords=c_cont, polar_coords=polar_coords_cont,
                                     orig_mask_filtered=orig_mask_filtered)
            vector1.append(v1)
            vector2.append(v2)

        # summarise contour scans
        max_ang_soil = summarise_scan(
            result=vector1,
            type="hull",  # get the maximum angle from convex hull
            ctype=ctype,
            regrown=reconstruct
        )
        share_veg, n_segments, prop_longest = summarise_scan(
            result=vector2,
            type="cont",  # ratio and the number of disconnected segments from contour
            ctype=ctype,
            regrown=reconstruct
        )

        # ==============================================================================================================

        # fix dilation level for the rest of the analyses
        if contour_type == 255:
            if reconstruct:
                cnt_dil, _ = dilate_contour(img=img, contour=ccc[i], ctype=contour_type, npix=18)
            else:
                cnt_dil, _ = dilate_contour(img=img, contour=ccc[i], ctype=contour_type, npix=36)
        elif contour_type == 125:
            cnt_dil, _ = dilate_contour(img=img, contour=ccc[i], ctype=contour_type, npix=18)

        # reshape contour data
        if training_coords is None:
            c = utils.flatten_contour_data([cnt_dil], asarray=False)
        else:
            if contour_type == 255:
                c = utils.flatten_contour_data([cnt_dil], asarray=False)

        polygon = np.array(c)

        # get the bounding box
        left = np.min(polygon, axis=0)
        right = np.max(polygon, axis=0)
        x = np.arange(math.ceil(left[0]), math.floor(right[0]) + 1)
        y = np.arange(math.ceil(left[1]), math.floor(right[1]) + 1)
        xv, yv = np.meshgrid(x, y, indexing='xy')
        points = np.hstack((xv.reshape((-1, 1)), yv.reshape((-1, 1))))

        # define empty masks (2D and 3D)
        zero_mask = np.zeros(img.shape, dtype=np.uint8)
        orig_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        # fill bounding box
        # this contains all pixel values of the bounding box: needs further processing
        for p in points:
            zero_mask[p[0], p[1]] = img[p[0], p[1], :]
            orig_mask[p[0], p[1]] = pp_mask[p[0], p[1]]

        # ==============================================================================================================

        # detect components and keep only the main component, removing smaller ones, neighbouring ones, and noise
        # get centroid coordinates for later use
        n_comps, output, stats, centroids = cv2.connectedComponentsWithStats(orig_mask, connectivity=8)
        sizes = list(stats[:, 4][1:])
        index = sizes.index(np.max(sizes))
        bin_num = np.uint8(np.where(output == index + 1, 1, 0))
        ctr = centroids[index + 1]

        # ==============================================================================================================

        # EXTRACT FEATURES FOR THE REMAINING COMPONENT
        # POSITION and SHAPE

        # features related to centroid position
        rowpoints = np.transpose(np.nonzero(rows))
        min_dist_to_row_ctr = np.amin(cdist([ctr], rowpoints, 'euclidean'))
        tgi_ctr = idx_map[int(ctr[1]), int(ctr[0])]

        # features to use from skimage.measure.regionprops
        label_img = measure.label(bin_num, connectivity=2)
        props = measure.regionprops(label_img)
        orientation = props[0].orientation
        eccentricity = props[0].eccentricity

        # ==============================================================================================================

        # custom features
        # get the component contour
        _, contour, _ = cv2.findContours(bin_num, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # measures of object elongation
        rect = cv2.minAreaRect(contour[0])
        bbox = np.int0(cv2.boxPoints(rect))
        (tl, tr, br, bl) = bbox
        xdist1 = tl[0] - tr[0]
        ydist1 = tl[1] - tr[1]
        length1 = math.sqrt(xdist1 * xdist1 + ydist1 * ydist1)
        xdist2 = tl[0] - bl[0]
        ydist2 = tl[1] - bl[1]
        length2 = math.sqrt(xdist2 * xdist2 + ydist2 * ydist2)
        long = max(length1, length2)
        short = min(length1, length2)
        try:
            elong = short / long
        except ZeroDivisionError:
            elong = np.NaN

        m = cv2.moments(bin_num)
        elong_m = elongation(m)

        # area ratio (component/convex hull)
        area = sizes[index]  # area of component
        points = utils.flatten_contour_data(contour, asarray=True)  # reshape point data
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
            area_ratio = area / area_hull

            # roundness
            vertices = hull.vertices.tolist() + [hull.vertices[0]]
            perimeter = np.sum([distance.euclidean(x, y) for x, y in zip(points[vertices], points[vertices][1:])])
            roundness = (4 * math.pi * area) / (perimeter * perimeter)

            # convexity defects
            hull = cv2.convexHull(contour[0], returnPoints=False)
            defects = cv2.convexityDefects(contour[0], hull)
            if defects is not None:
                n_convdefs = len(defects)
            else:
                n_convdefs = 0
        except:
            area_ratio = np.NaN
            roundness = np.NaN
            n_convdefs = np.NaN

        # ==============================================================================================================

        # coordinates of pixels making up the plant object
        x, y = np.where(output == index + 1)
        coords = []
        for a, b in zip(x, y):
            coords.append([a, b])
        polygon = np.array(coords)

        # new bounding box for the kept component
        left = np.min(polygon, axis=0)
        right = np.max(polygon, axis=0)
        x = np.arange(math.ceil(left[0]), math.floor(right[0]) + 1)
        y = np.arange(math.ceil(left[1]), math.floor(right[1]) + 1)
        xv, yv = np.meshgrid(x, y, indexing='xy')
        points = np.hstack((xv.reshape((-1, 1)), yv.reshape((-1, 1))))

        # define empty masks (2D and 3D)
        zero_mask = np.zeros(img.shape, dtype=np.uint8)
        orig_mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # fill for the coordinates containing plant
        for p in polygon:
            zero_mask[p[0], p[1]] = img[p[0], p[1], :]
            orig_mask[p[0], p[1]] = orig_mask_filtered[p[0], p[1]]

        # ==============================================================================================================

        # get the number of holes and their relative total area
        # Standard measure seems to be the Euler number. But here, number of connected components is always 1.
        _, contours_, hier_ = cv2.findContours(orig_mask, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)
        holes = [contours_[i] for i in range(len(contours_)) if hier_[0][i][3] >= 0]
        n_holes = len(holes)
        area_holes = []
        for hole in holes:
            area_ = cv2.contourArea(hole)
            area_holes.append(area_)
        area_tot = np.sum(area_holes)
        area_ratio_holes = area_tot / area

        # ==============================================================================================================

        _, contour_, _ = cv2.findContours(bin_num, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # identify pixels lying within the contour
        # replace with actual RGB values or with binary vegetation/background value
        path = mpl.path.Path(c)
        mask = path.contains_points(points, radius=0)
        mask.shape = xv.shape

        # wtf?! path.contains_points returns everything head-over!
        mask = np.swapaxes(mask.reshape(mask.shape[0], mask.shape[1]), 0, 1)

        polygon_coords = np.argwhere(mask == True)
        # mask for RGB image
        final_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        # mask for binary mask
        final_bin_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for pixel in polygon_coords:
            # replace by RGB values
            final_mask[pixel[0], pixel[1]] = zero_mask[left[0] + pixel[0], left[1] + pixel[1], :]
            # replace with binary pattern
            final_bin_mask[pixel[0], pixel[1]] = orig_mask[left[0] + pixel[0], left[1] + pixel[1]]

        # ==============================================================================================================

        patch = final_mask

        # convert pixels to gray scale
        graypatch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)

        # calculate the grey level covariance matrix and extract Haralick texture features
        glcm = greycomatrix(graypatch, distances=[1, 2, 3, 4, 6, 8, 12], angles=[0, np.pi / 2], levels=256,
                            symmetric=True, normed=True)

        # zero intensity value observed for background pixels,
        # obtain the GLCM of the ROI by simply discarding the first line
        # and the first column of the full image's GLCM.
        glcm_br = glcm[1:, 1:, :, :]
        try:
            glcm_br_norm = np.true_divide(glcm_br, glcm_br.sum(axis=(0, 1)))
            x = greycoprops(glcm_br_norm, prop='dissimilarity')
            x = x.flatten()
        except:
            x = np.empty((14,))
            x[:] = np.NaN

        gcprops = []
        for prop in range(len(x)):
            d = {f'greycoprop_{prop + 1}': x[prop]}
            gcprops.append(d)

        greycopreds = {}
        for prop in gcprops:
            greycopreds.update(prop)

        # get contours of original components
        try:
            mask_component = morphology.area_closing(final_bin_mask, area_threshold=50)
        # for very small components on image edge
        except ValueError:
            mask_component = final_bin_mask
        mask_component_erode = morphology.erosion(mask_component, selem=np.ones((2, 2), np.uint8))
        coords = np.where(mask_component_erode == 0)

        # calculate entropy
        img_ent = Entropy(graypatch, morphology.disk(2))

        # remove border region
        img_ent_def = np.where(mask_component_erode == 255, img_ent, 0)

        entr_value = []
        for pixel in polygon_coords:
            entr_value.append(img_ent_def[pixel[0]][pixel[1]])
        entr_value = [i for i in entr_value if i != 0]

        entr_mean = np.mean(entr_value)
        entr_median = np.median(entr_value)
        entr_kurt = kurtosis(entr_value)
        entr_skew = skew(entr_value)
        entr_entr = entropy(entr_value)

        # ==============================================================================================================

        # COLOR PROPERTIES

        color_spaces, descriptors, descriptor_names = ImageFunctions.get_color_spaces(img)

        # calculate descriptive statistics of color descriptors
        col_preds = []
        for descriptor in range(0, descriptors.shape[2]):
            desc_name = descriptor_names[descriptor]
            desc = descriptors[:, :, descriptor]
            orig_mask = np.zeros(img.shape[:2], dtype=np.float32)
            value = []
            for p in polygon:
                orig_mask[p[0], p[1]] = desc[p[0], p[1]]
                value.append(desc[p[0], p[1]])

            # mask for values
            final_bin_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.float32)
            for pixel in polygon_coords:
                # replace with binary pattern
                final_bin_mask[pixel[0], pixel[1]] = orig_mask[left[0] + pixel[0], left[1] + pixel[1]]

            col_value = []
            for pixel in polygon:
                col_value.append(desc[pixel[0]][pixel[1]])

            mn = np.mean(col_value)
            md = np.median(col_value)
            kt = kurtosis(col_value)
            sk = skew(col_value)
            en = entropy(col_value)

            d = {f'{desc_name}_mean': mn, f'{desc_name}_median': md, f'{desc_name}_kurt': kt,
                 f'{desc_name}_skew': sk, f'{desc_name}_entr': en}

            col_preds.append(d)

        colorpreds = {}
        for col in col_preds:
            colorpreds.update(col)

        # output label for training positions
        if training_coords is None:
            XY = ({'ctype': ctype,
                   "ctr_x": ctr[1], "ctr_y": ctr[0],
                   'min_dist_to_row_ctr': min_dist_to_row_ctr, 'tgi_ctr': tgi_ctr,
                   'share_veg': share_veg, 'n_segments': n_segments, 'max_ang_soil': max_ang_soil,
                   'prop_longest': prop_longest,
                   'n_holes': n_holes, 'area_ratio_holes': area_ratio_holes,
                   'area': area,
                   'elong': elong, 'elong_m': elong_m,
                   'orientation': orientation,
                   'eccentricity': eccentricity,
                   'n_convdefs': n_convdefs, 'area_ratio': area_ratio,
                   'roundness': roundness,
                   'entr_mean': entr_mean, 'entr_median': entr_median, 'entr_kurt': entr_kurt,
                   'entr_skew': entr_skew, 'entr_entr': entr_entr})
        else:
            XY = ({'class_label': labs[i],
                   'ctype': ctype,
                   "ctr_x": ctr[1], "ctr_y": ctr[0],
                   'min_dist_to_row_ctr': min_dist_to_row_ctr, 'tgi_ctr': tgi_ctr,
                   'share_veg': share_veg, 'n_segments': n_segments, 'max_ang_soil': max_ang_soil,
                   'prop_longest': prop_longest,
                   'n_holes': n_holes, 'area_ratio_holes': area_ratio_holes,
                   'area': area,
                   'elong': elong, 'elong_m': elong_m,
                   'orientation': orientation,
                   'eccentricity': eccentricity,
                   'n_convdefs': n_convdefs, 'area_ratio': area_ratio,
                   'roundness': roundness,
                   'entr_mean': entr_mean, 'entr_median': entr_median, 'entr_kurt': entr_kurt,
                   'entr_skew': entr_skew, 'entr_entr': entr_entr})

        XY = {**XY, **greycopreds, **colorpreds}
        preds.append(XY)

    if training_coords is None:
        return outputr, preds
    else:
        return preds


def extract_img_features(path_rowmask, picture_type):
    """
    Extracts the mean tgi value for detected crop rows and for the inter-row space from full images.
    Must use different sources of information, depending on the type of image and on the date. HORRIBLE!!!!
    These files should all be regenerated in the exact same manner, for handheld images and for UAV images.
    :param path_rowmask: path (character string)
    :param picture_type: picture type (character string)
    :return: the means row tgi and the mean interspace tgi; the row mask
    """
    if picture_type == "Handheld":
        rows = mpimg.imread(path_rowmask)
        path_row_json = re.sub(".tif", ".json", path_rowmask)

        with open(path_row_json) as json_file:
            data = json.load(json_file)
        vs_row = data['plantindex_row']['idx_row']
        vs_is = data['plantindex_interspace']['idx_is_mid']
    else:
        # Extraworscht für de Herbifly Chröppelrechner
        # rows = mpimg.imread(path_rowmask)[1216:2432, 1824:3648]
        rows = imageio.imread(path_rowmask)[1216:2432, 1824:3648]
        rows = np.where(rows, 0, 1)

        with open(path_rowmask, 'rb') as f:
            tags = exifread.process_file(f)
        v = tags['Image ImageDescription'].values
        # not identical across all images !?!
        try:
            vs_row = json.loads(v)['idx_row']
            vs_is = json.loads(v)['idx_is']
        except TypeError:
            try:
                vs_row = json.loads(v)[4]['idx_row']
                vs_is = json.loads(v)[4]['idx_is']
            except KeyError:
                path_row_json = re.sub(".tif", ".json", path_rowmask)
                with open(path_row_json) as json_file:
                    data = json.load(json_file)
                vs_row = data['plantindex_row'][4]['idx_row']
                vs_is = data['plantindex_interspace'][4]['idx_is_mid']

    mean_row_tgi = np.asarray(vs_row).mean()
    mean_is_tgi = np.asarray(vs_is).mean()
    try:
        rows, _, _, _ = cv2.split(rows)
    # Extraworscht für de Herbifly Chröppelrechner
    except ValueError:
        rows = rows

    return mean_row_tgi, mean_is_tgi, rows


def predict_obj_class(img, ppmask, pic_name, X, path_current_prediction,
                      path_trained_component_classification_model,
                      size_threshold):

    # Not optimal:  feature data needs to be matched with the components via their centroid positions
    # Probably due to slightly different shape of components (erosion in feature extraction),
    # Messing up the object ordering

    # fill holes in objects to avoid nested contours
    _, contours_rec, _ = cv2.findContours(ppmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask_no_holes = np.zeros(ppmask.shape, dtype="uint8")
    cv2.drawContours(mask_no_holes, contours_rec, -1, 255, -1)
    _, labels, _, centroids = cv2.connectedComponentsWithStats(mask_no_holes, connectivity=8)

    # get centroids from feature data
    cX = X["ctr_x"].to_numpy(dtype=int)
    cY = X["ctr_y"].to_numpy(dtype=int)
    # get centroids from the post-processed mask objects
    ctrds_ = list(zip(cY, cX))
    ctrds = [list(map(int, i)) for i in centroids[1:]]

    # find the centroids with the smallest distance from list and extract its index
    # this is then used to match feature data with the object labels from the post-processed mask
    inds = []
    for i in range(len(ctrds_)):
        index = np.argmin(cdist([ctrds_[i]], ctrds, 'euclidean'))
        inds.append(index)

    # remove rows with missing predictor values
    ddd = X.dropna()
    # remove centroid position - not a predictor
    ddd = ddd.drop(["ctr_x", "ctr_y"], axis=1)
    # convert to array
    ddd = np.asarray(ddd)

    # load pre-trained rf classification model
    print('>>Classifying components...')
    with open(path_trained_component_classification_model, 'rb') as file:
        model = pickle.load(file)

    # create prediction using pre-trained model
    lab = model.predict(ddd)

    # get indices of large components
    # these are classified as wheat ex-post
    index = X.index
    index_large_comps = index[X["area"] > size_threshold]

    # fill up label vector with NAs where no prediction could be made (missing values)
    idx = [index for index, row in X.iterrows() if row.isnull().any()]
    vec = list(range(len(X)))
    vec_labs = np.setdiff1d(vec, idx).tolist()
    labs = []
    counter = 0
    for i in vec:
        if i in vec_labs:
            l = lab[counter]
            counter += 1
        else:
            l = "wheat"
        labs.append(l)

    for index in index_large_comps:
        # print(labs[index])
        labs[index] = "wheat"

    # set all pixel values belonging to wheat to zero
    filtered_comps = np.where(np.asarray(labs) == 'wheat')[0]
    mask_filtered = copy.copy(ppmask)
    # remove all wheat components from mask
    for i in filtered_comps:
        mask_filtered[labels == inds[i] + 1] = 0
    mask_filtered = mask_filtered.astype("uint8")

    # Draw contour classified as wheat onto original image
    weeds = copy.copy(img)
    _, all_contours, _ = cv2.findContours(ppmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    _, contours, _ = cv2.findContours(mask_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # weeds = cv2.drawContours(weeds, all_contours, -1, (0, 0, 255), -1)
    weeds = cv2.drawContours(weeds, contours, -1, (255, 0, 0), -1)

    # Save image
    path_pred_img = "{path_current_pred}/{pic_n}_predicted_image.tif".format(
        path_current_pred=path_current_prediction, pic_n=pic_name)
    imageio.imwrite(path_pred_img, weeds)
    # save mask with weeds marked
    path_pred_mask = "{path_current_pred}/{pic_n}_predicted_mask.tif".format(
        path_current_pred=path_current_prediction, pic_n=pic_name)
    imageio.imwrite(path_pred_mask, mask_filtered)

    return weeds, mask_filtered


def iterate_training_images(path_to_files, img_dir, plot_pos, reconstruct):

    # get all files (each containing all training positions for one image)
    files = glob.glob(f'{path_to_files}/training_coords/*.csv')
    print(f'-Found {len(files)} files')

    # get training coordinates
    coords = []
    for file in files:
        c = pd.read_csv(file)
        coords.append(c)

    # get image names from the .csv
    images = list(pd.unique(pd.concat(coords)["img_id"]))

    # iterate over images
    for i, im in enumerate(images):

        # read image and masks
        img = imageio.imread(re.sub(".tif", "_tile_orig.tif", images[i]))
        mask = imageio.imread(images[i])
        mask_filtered = cv2.medianBlur(mask, 5)
        img_cmask = imageio.imread(re.sub(".tif", "_mask_pp.tif", images[i]))
        img_cmaskall_rec = imageio.imread(re.sub(".tif", "_mask_pp_all_rec.tif", images[i]))

        # get/set paths
        nameidx = [im.split('/')[i] for i in [6, 8, 9]]
        nameidx.append(utils.get_farmer_region(nameidx[0]))
        nameidx[2] = re.sub(".tif", "", nameidx[2])
        trcname = nameidx[2]
        path_rowsprof = str(f'{img_dir}/{nameidx[3]}/{nameidx[0]}/Handheld/{nameidx[1]}/idx_map_combine_py5fullresTGI_tile1x1/{nameidx[2]}_idx_combine.png')
        path_rows = str(f'{img_dir}/{nameidx[3]}/{nameidx[0]}/Handheld/{nameidx[1]}/rowmask_py5fullresTGI_tile1x1/{nameidx[2]}_rowmask.tif')
        path_tgi_vals = str(f'{img_dir}/{nameidx[3]}/{nameidx[0]}/Handheld/{nameidx[1]}/rowmask_py5fullresTGI_tile1x1/{nameidx[2]}_rowmask.json')

        # iter
        print(f'--{i+1}/{len(files)} : {trcname}')

        # read crop row information
        rows = mpimg.imread(path_rows)
        rows, _, _, _ = cv2.split(rows)
        idx_map = mpimg.imread(path_rowsprof)
        with open(path_tgi_vals) as json_file:
            data = json.load(json_file)

        # get image-related features from exif data
        vs_row = data['plantindex_row']['idx_row']
        vs_is = data['plantindex_interspace']['idx_is_mid']
        mean_row_tgi = np.asarray(vs_row).mean()
        mean_is_tgi = np.asarray(vs_is).mean()

        # get training positions in list
        df = coords[i]
        training_coords = []
        for row in df.iterrows():
            x = row[1]['x']
            y = row[1]['y']
            set = row[1]['set']
            cnt_type = row[1]["cnt_type"]
            training_coords.append({'x': x, 'y': y, 'set': set, 'cnt_type': cnt_type})

        pos_out_dir = f'{path_to_files}/training_coords/pos/'
        Path(pos_out_dir).mkdir(parents=True, exist_ok=True)

        # if training positions should be marked on images and masks
        if plot_pos:

            # set paths
            path_check_img_out = f'{pos_out_dir}/{trcname}.tif'
            path_check_mask_out = f'{pos_out_dir}/{trcname}_mask.tif'
            path_check_eroded_mask_out = f'{pos_out_dir}/{trcname}_eroded_mask.tif'

            # mark positions
            img_trpos, pp_mask_trpos, eroded_mask_trpos = mark_training_positions(
                img=img,
                eroded_mask=img_cmask,
                rec_mask=img_cmaskall_rec,
                coords=training_coords
            )
            # save images
            imageio.imwrite(path_check_img_out, img_trpos)
            imageio.imwrite(path_check_mask_out, pp_mask_trpos)
            imageio.imwrite(path_check_eroded_mask_out, eroded_mask_trpos)

        # select mask to use and set dir name
        if reconstruct:
            pp_mask = img_cmaskall_rec
            out_folder_name = "reconstruct"
        else:
            pp_mask = img_cmask
            out_folder_name = "eroded"

        # output directory
        path_out = f'{path_to_files}/training_data/{out_folder_name}'
        Path(path_out).mkdir(parents=True, exist_ok=True)

        filename_out = f'{path_out}/{trcname}.csv'
        if Path(filename_out).exists():
            print("Features already exist. Skipping feature extraction.")
            continue
        else:
            traindat = extract_obj_features(img=img,
                                            picture_type="Handheld",
                                            orig_mask_filtered=mask_filtered,
                                            pp_mask=pp_mask,
                                            rows=rows,
                                            idx_map=idx_map,
                                            reconstruct=reconstruct,
                                            coords=training_coords)

            df = pd.DataFrame(traindat)
            # add image-based predictors
            df['mean_row_tgi'] = mean_row_tgi
            df['mean_is_tgi'] = mean_is_tgi
            df.to_csv(filename_out, index=False)


def iterate_training_images_uav(path_to_files, img_dir, plot_pos, features):
    # get all files (each containing all training positions for one image)
    files = glob.glob(f'{path_to_files}/training_coords/*.csv')
    print(f'-Found {len(files)} files')

    # get training coordinates
    coords = []
    for file in files:
        c = pd.read_csv(file)
        coords.append(c)

    # get image names from the .csv
    images = list(pd.unique(pd.concat(coords)["img_id"]))

    # iterate over images
    for i, im in enumerate(images):

        # read image and masks
        img = imageio.imread(re.sub(".tif", "_tile_orig.tif", images[i]))
        mask = imageio.imread(images[i])
        pp_mask = imageio.imread(re.sub(".tif", "_mask_pp.tif", images[i]))
        mask_filtered = np.where(pp_mask==0, pp_mask, 255)  # different from handheld ! Original Mask is not used here.

        # get/set paths
        nameidx = [im.split('/')[i] for i in [6, 8, 9]]
        nameidx.append(utils.get_farmer_region(nameidx[0]))
        nameidx[2] = re.sub(".tif", "", nameidx[2])
        trcname = "_".join(nameidx[0:3])
        path_rows = str(f'{img_dir}/{nameidx[3]}/{nameidx[0]}/10m/{nameidx[1]}/rowmask_py0TGI_tile3x3/{nameidx[2]}_rowmask.tif')
        path_rowsprof = str(f'{img_dir}/{nameidx[3]}/{nameidx[0]}/10m/{nameidx[1]}/idx_map_combine_py0TGI_tile3x3/{nameidx[2]}_idx_combine.png')

        # iter
        print(f'--{i + 1}/{len(files)} : {trcname}')

        mean_row_tgi, mean_is_tgi, rows = extract_img_features(path_rowmask=path_rows,
                                                               picture_type="10m")
        idx_map = imageio.imread(path_rowsprof)[1216:2432, 1824:3648]

        # get training positions in list
        df = coords[i]
        training_coords = []
        for row in df.iterrows():
            x = row[1]['x']
            y = row[1]['y']
            set = row[1]['set']
            cnt_type = row[1]["cnt_type"]
            training_coords.append({'x': x, 'y': y, 'set': set, 'cnt_type': cnt_type})

        pos_out_dir = f'{path_to_files}/training_coords/pos/'
        Path(pos_out_dir).mkdir(parents=True, exist_ok=True)

        # if training positions should be marked on images and masks
        if plot_pos:
            # set paths
            path_check_img_out = f'{pos_out_dir}/{trcname}.tif'
            path_check_mask_out = f'{pos_out_dir}/{trcname}_mask.tif'

            # mark positions
            img_trpos, _, mask_trpos = mark_training_positions(img=img,
                                                               eroded_mask=pp_mask,
                                                               rec_mask=None,
                                                               coords=training_coords)
            # save images
            imageio.imwrite(path_check_img_out, img_trpos)
            imageio.imwrite(path_check_mask_out, mask_trpos)

        # select mask to use and set dir name
        if features == "subset":
            out_folder_name = "subset"
        elif features == "all":
            out_folder_name = "all"

        # output directory
        path_out = f'{path_to_files}/training_data/{out_folder_name}'
        Path(path_out).mkdir(parents=True, exist_ok=True)

        filename_out = f'{path_out}/{trcname}.csv'
        if Path(filename_out).exists():
            print("Features already exist. Skipping feature extraction.")
            continue
        else:
            if features == "all":
                traindat = extract_obj_features(img=img,
                                                picture_type="10m",
                                                orig_mask_filtered=mask_filtered,
                                                pp_mask=pp_mask,
                                                rows=rows,
                                                idx_map=idx_map,
                                                reconstruct=True,
                                                training_coords=training_coords)
            elif features == "subset":
                traindat = extract_obj_features_subset(img=img,
                                                       mask=pp_mask,
                                                       training_coords=training_coords,
                                                       rows=rows,
                                                       idx_map=idx_map)

            df = pd.DataFrame(traindat)
            # add image-based predictors
            df['mean_row_tgi'] = mean_row_tgi
            df['mean_is_tgi'] = mean_is_tgi
            df.to_csv(filename_out, index=False)


# ======================================================================================================================

# deprecated

def get_object_descriptors_hh(img, orig_mask_filtered, pp_mask, coords, rows, idx_map, reconstruct):
    # extract labelled objects' contours, corresponding type info and label
    print("---extracting labelled contours...")
    ccc, labelled_contours, cnt_type_reordered, labs = grab_contours(pp_mask=pp_mask, coords=coords)

    # get predictors for each labelled contour
    preds = []
    print("---extracting features...")
    for i, c in enumerate(labelled_contours):

        # get dilatation factors depending on contour type and post-processing procedure
        if reconstruct:
            if cnt_type_reordered[i] == 255:
                ctype = 'eroded'
                dfact = list(range(2, 14))
            elif cnt_type_reordered[i] == 125:
                ctype = 'original'
                dfact = list(range(2, 14))
        else:
            if cnt_type_reordered[i] == 255:
                ctype = 'eroded'
                dfact = list(range(18, 30))
            elif cnt_type_reordered[i] == 125:
                ctype = 'original'
                dfact = list(range(2, 14))

        # iterate over different levels of dilation
        vector1 = []
        vector2 = []
        for k in range(len(dfact)):
            # dilate contour
            cont, hull = dilate_contour(img=img, contour=ccc[i], ctype=cnt_type_reordered[i], npix=dfact[k])
            # get contour centroid
            ctr_hull = get_centroid(contour=hull)
            ctr_cont = get_centroid(contour=cont[0])
            # get coordinates of contour points and hull points
            c_cont = get_cont_coordinates(contour=cont[0], img=img)
            c_hull = get_cont_coordinates(contour=hull, img=img)
            # transform from cartesian to polar coordinates
            polar_coords_cont = utils.cart2pol(x=c_cont[1], y=c_cont[0], ctr=ctr_cont)
            polar_coords_hull = utils.cart2pol(x=c_hull[1], y=c_hull[0], ctr=ctr_hull)

            # get pixel values on hull
            v1 = scan_format_contour(coords=c_hull, polar_coords=polar_coords_hull,
                                     orig_mask_filtered=orig_mask_filtered)
            # get pixel values contour
            v2 = scan_format_contour(coords=c_cont, polar_coords=polar_coords_cont,
                                     orig_mask_filtered=orig_mask_filtered)
            vector1.append(v1)
            vector2.append(v2)

        # summarise contour scans
        max_ang_soil = summarise_scan(
            result=vector1,
            type="hull",
            ctype=ctype,
            regrown=reconstruct  # get the maximum angle from convex hull
        )
        share_veg, n_segments, prop_longest = summarise_scan(
            result=vector2,
            type="cont",
            ctype=ctype,
            regrown=reconstruct  # ratio and the number of disconnected segments from contour
        )

        # # get type of contour
        # if cnt_type_reordered[i] == 255:
        #     ctype = "eroded"
        # elif cnt_type_reordered[i] == 125:
        #     ctype = "original"
        #
        # dil_factors_large = list(range(18, 30))
        # dil_factors_small = list(range(2, 14))
        #
        # # get different dilatation of contours and sample
        # vector = []
        # blank0 = np.zeros(img.shape[:2], dtype=np.uint8)
        # img2 = copy.copy(img)
        # for k in range(len(dil_factors_large)):
        #
        #     # define empty masks to draw contours and hulls onto
        #     blank = np.zeros(img.shape[:2], dtype=np.uint8)
        #     blank_a = copy.copy(blank)
        #     blank_b = copy.copy(blank)
        #
        #     # draw thick contours (dilatation)
        #     if cnt_type_reordered[i] == 255:
        #         cv2.drawContours(blank_a, ccc[i], -1, 255, dil_factors_large[k])
        #     elif cnt_type_reordered[i] == 125:
        #         cv2.drawContours(blank_a, ccc[i], -1, 255, dil_factors_small[k])
        #     cv2.fillPoly(blank_a, pts=[ccc[i]], color=255)
        #     # this is needed for contours lying on the image edge!!
        #     res0 = cv2.medianBlur(blank_a, 5)
        #
        #     # get the dilated contour
        #     cnt_dil, _ = cv2.findContours(res0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #
        #     # get centroid of contour
        #     M = cv2.moments(cnt_dil[0])
        #     cX = int(M["m10"] / M["m00"])
        #     cY = int(M["m01"] / M["m00"])
        #     ctr = [cY, cX]
        #     cv2.circle(img2, (cX, cY), 5, (255, 0, 0), -1)
        #
        #     # convex hull
        #     c_hull = cv2.convexHull(cnt_dil[0], False)
        #     cv2.drawContours(blank_b, [c_hull], -1, 75, 1)
        #     cv2.drawContours(img2, [c_hull], -1, (0, 0, 255), 1)
        #     cv2.drawContours(blank_a, cnt_dil, -1, 125, 1)
        #     cv2.drawContours(img2, cnt_dil, -1, (0, 255, 0), 1)
        #
        #     # extract coordinates
        #     coords_cont = np.where(blank_a == 125)
        #     coords_chull = np.where(blank_b == 75)
        #
        #     coords = coords_cont
        #
        #     # get centroid of contour
        #     M = cv2.moments(c_hull)
        #     cX = int(M["m10"] / M["m00"])
        #     cY = int(M["m01"] / M["m00"])
        #     ctr = [cY, cX]
        #
        #     # transform from cartesian to polar coordinates
        #     c_pol = utils.cart2pol(x=coords[1], y=coords[0], ctr=ctr)
        #
        #     # sort according to angle
        #     cxx = np.argsort(c_pol[1])
        #     ang = c_pol[1][cxx]
        #     pos = c_pol[0][cxx]
        #     # get contour pixel values and reorder
        #     cpix_vals = np.bitwise_not(orig_mask_filtered)[coords]
        #     vals = cpix_vals[cxx]
        #     # interpolate to specific angles
        #     newang = np.arange(start=-math.pi, stop=math.pi, step=math.pi/1080)
        #     vv = np.interp(x=newang, xp=ang, fp=vals)
        #     vector.append(vv)
        #
        #     # back to cartesian coordinates
        #     ccart = utils.pol2cart(rho=pos, phi=ang, ctr=ctr)
        #     blank0[ccart] = cv2.bitwise_not(orig_mask_filtered)[ccart]
        #
        # v = np.column_stack(vector)  # stack columns
        # v_ = v.max(axis=1).astype(int)  # get the maximum per row
        # # find longest consecutive soil segment
        # zeros = np.where(v_ == 0)[0]  # find index of plant pixel
        # tot_length_cont = len(v_)  # total length of the contour
        # lengths = np.ediff1d(zeros)  # get vector distance between plant pixels to obtain length of soil segment
        #
        # # if the soil segments overlaps with the start/end of the contour
        # if not zeros[0] == 0:
        #     length = tot_length_cont - zeros[-1] + zeros[0]
        #     lengths = lengths.tolist()
        #     lengths.append(length)
        #     lengths = np.asarray(lengths)
        #
        # soil_segments = len(np.where(lengths > 1)[0])  # get the number of disconnected soil segments
        # max_length = max(lengths)  # get the longest soil segment
        # # proportion of the longest soil segment
        # prop_longest = max_length/tot_length_cont
        # # get angle of the longest soil segment
        # index = np.where(lengths == max_length)[0][0]
        # lower = zeros[index]  # index of the first soil pixel
        #
        # # if the soil segments overlaps with the start/end of the contour
        # try:
        #     upper = zeros[index + 1]  # index of the last soil pixel
        #     angs = newang[lower:upper]
        # except:
        #     angs = newang[lower:len(newang)].tolist() + newang[0:zeros[0]].tolist()
        #     angs = np.asarray(angs)
        #
        # # get the distance as an angle
        # angs = 180 * angs / math.pi
        # angs = angs.astype(int)
        # max_ang_soil = np.max(angs) - np.min(angs)
        # # get ratio veg/soil
        # ratio = len(zeros)/tot_length_cont
        # # get number of plant segments segments
        # nonzeros = np.where(v_ != 0)[0]
        # lengths = np.ediff1d(nonzeros)
        # plant_segments = len(np.where(lengths > 1)[0])
        # # get n_segments as the maximum of soil and plant segments (depends on where the contour starts!)
        # n_segements = max(soil_segments, plant_segments)
        #
        # tot_length_cont = len(v_)
        # lengths = np.ediff1d(zeros)
        # n_segments = len(np.where(lengths > 1)[0])
        # max_length = max(lengths)
        # # proportion of the longest segment
        # prop_longest = max_length/tot_length_cont
        # index = np.where(lengths == max_length)[0][0]
        # lower = zeros[index]
        # upper = zeros[index+1]
        # # as angle
        # angs = newang[lower:upper]
        # angs = 180 * angs / math.pi
        # angs = angs.astype(int)
        # max_ang_soil = np.max(angs) - np.min(angs)
        # # get ratio veg/soil
        # ratio = len(zeros)/tot_length_cont
        #
        # # Plot result
        # fig, axs = plt.subplots(1, 4, sharex=True, sharey=True)
        # # Show RGB and segmentation mask
        # axs[0].imshow(img2)
        # axs[0].set_title('img')
        # axs[1].imshow(orig_mask_filtered)
        # axs[1].set_title('orig_mask')
        # axs[2].imshow(pp_mask)
        # axs[2].set_title('post-processed mask')
        # axs[3].imshow(blank0)
        # axs[3].set_title('contour scan')
        # plt.show(block=True)
        #
        # print(i)

        # ==============================================================================================================

        # fix dilation level for the rest of the analyses
        # should be relatively insensitive to this...
        if cnt_type_reordered[i] == 255:
            if reconstruct:
                cnt_dil, _ = dilate_contour(img=img, contour=ccc[i], ctype=cnt_type_reordered[i], npix=18)
            else:
                cnt_dil, _ = dilate_contour(img=img, contour=ccc[i], ctype=cnt_type_reordered[i], npix=36)
        elif cnt_type_reordered[i] == 125:
            cnt_dil, _ = dilate_contour(img=img, contour=ccc[i], ctype=cnt_type_reordered[i], npix=18)

        # reshape contour data
        if cnt_type_reordered[i] == 255:
            c = utils.flatten_contour_data(cnt_dil, asarray=False)

        polygon = np.array(c)

        # get the bounding box
        left = np.min(polygon, axis=0)
        right = np.max(polygon, axis=0)
        x = np.arange(math.ceil(left[0]), math.floor(right[0]) + 1)
        y = np.arange(math.ceil(left[1]), math.floor(right[1]) + 1)
        xv, yv = np.meshgrid(x, y, indexing='xy')
        points = np.hstack((xv.reshape((-1, 1)), yv.reshape((-1, 1))))

        # define empty masks (2D and 3D)
        zero_mask = np.zeros(img.shape, dtype=np.uint8)
        orig_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        # fill bounding box
        # this contains all pixel values of the bounding box: needs further processing
        for p in points:
            zero_mask[p[0], p[1]] = img[p[0], p[1], :]
            orig_mask[p[0], p[1]] = pp_mask[p[0], p[1]]

        # ==================================================================================================================

        # detect components and keep only the main component, removing smaller ones, neighbouring ones, and noise
        # get centroid coordinates for later use
        n_comps, output, stats, centroids = cv2.connectedComponentsWithStats(orig_mask, connectivity=8)
        sizes = list(stats[:, 4][1:])
        index = sizes.index(np.max(sizes))
        bin = np.where(output == index + 1, True, False)
        bin_num = np.uint8(np.where(output == index + 1, 1, 0))
        ctr = centroids[index + 1]
        # done

        # # Plot result
        # fig, axs = plt.subplots(1, 4, sharex=True, sharey=True)
        # # Show RGB and segmentation mask
        # # axs[0].imshow(zero_mask)
        # axs[0].imshow(bin_num)
        # axs[0].set_title('contour')
        # axs[1].imshow(orig_mask)
        # axs[1].set_title('mask')
        # axs[2].imshow(orig_mask_filtered)
        # axs[2].set_title('mask')
        # axs[3].imshow(pp_mask)
        # axs[3].set_title('mask')
        # plt.show(block=True)

        # ==============================================================================================================

        # EXTRACT FEATURES FOR THE REMAINING COMPONENT
        # POSITION and SHAPE

        # features related to centroid position
        rowpoints = np.transpose(np.nonzero(rows))
        min_dist_to_row_ctr = np.amin(cdist([ctr], rowpoints, 'euclidean'))
        tgi_ctr = idx_map[int(ctr[1]), int(ctr[0])]

        # features to use from skimage.measure.regionprops
        label_img = measure.label(bin_num, connectivity=2)
        props = measure.regionprops(label_img)
        orientation = props[0].orientation
        eccentricity = props[0].eccentricity

        # ==============================================================================================================

        # custom features
        # get the component contour
        contour, _ = cv2.findContours(bin_num, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # measures of object elongation
        rect = cv2.minAreaRect(contour[0])
        bbox = np.int0(cv2.boxPoints(rect))
        (tl, tr, br, bl) = bbox
        xdist1 = tl[0] - tr[0]
        ydist1 = tl[1] - tr[1]
        length1 = math.sqrt(xdist1 * xdist1 + ydist1 * ydist1)
        xdist2 = tl[0] - bl[0]
        ydist2 = tl[1] - bl[1]
        length2 = math.sqrt(xdist2 * xdist2 + ydist2 * ydist2)
        long = max(length1, length2)
        short = min(length1, length2)
        elong = short / long

        m = cv2.moments(bin_num)
        elong_m = elongation(m)

        # area ratio (component/convex hull)
        area = sizes[index]  # area of component
        points = utils.flatten_contour_data(contour, asarray=True)  # reshape point data
        hull = ConvexHull(points)
        area_hull = hull.volume
        area_ratio = area / area_hull

        # roundness
        vertices = hull.vertices.tolist() + [hull.vertices[0]]
        perimeter = np.sum([distance.euclidean(x, y) for x, y in zip(points[vertices], points[vertices][1:])])
        roundness = (4 * math.pi * area) / (perimeter * perimeter)

        # convexity defects
        hull = cv2.convexHull(contour[0], returnPoints=False)
        defects = cv2.convexityDefects(contour[0], hull)
        if defects is not None:
            n_convdefs = len(defects)
        else:
            n_convdefs = 0

        # ==================================================================================================================

        # coordinates of pixels making up the plant object
        x, y = np.where(output == index + 1)
        coords = []
        for a, b in zip(x, y):
            coords.append([a, b])
        polygon = np.array(coords)

        # new bounding box for the kept component
        left = np.min(polygon, axis=0)
        right = np.max(polygon, axis=0)
        x = np.arange(math.ceil(left[0]), math.floor(right[0]) + 1)
        y = np.arange(math.ceil(left[1]), math.floor(right[1]) + 1)
        xv, yv = np.meshgrid(x, y, indexing='xy')
        points = np.hstack((xv.reshape((-1, 1)), yv.reshape((-1, 1))))

        # define empty masks (2D and 3D)
        zero_mask = np.zeros(img.shape, dtype=np.uint8)
        orig_mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # fill for the coordinates containing plant
        for p in polygon:
            zero_mask[p[0], p[1]] = img[p[0], p[1], :]
            orig_mask[p[0], p[1]] = orig_mask_filtered[p[0], p[1]]

        # # Plot result
        # fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
        # # Show RGB and segmentation mask
        # axs[0].imshow(zero_mask)
        # axs[0].set_title('contour')
        # axs[1].imshow(orig_mask)
        # axs[1].set_title('mask')
        # axs[2].imshow(orig_mask_filtered)
        # axs[2].set_title('mask')
        # plt.show(block=True)

        # ==============================================================================================================

        # get the number of holes and their relative total area
        # Standard measure seems to be the Euler number. But here, number of connected components is always 1.
        contours_, hier_ = cv2.findContours(orig_mask, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)
        holes = [contours_[i] for i in range(len(contours_)) if hier_[0][i][3] >= 0]
        n_holes = len(holes)
        area_holes = []
        for hole in holes:
            area_ = cv2.contourArea(hole)
            area_holes.append(area_)
        area_tot = np.sum(area_holes)
        area_ratio_holes = area_tot / area

        # ==============================================================================================================

        contour_, _ = cv2.findContours(bin_num, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # identify pixels lying within the contour
        # replace with actual RGB values or with binary vegetation/background value
        path = mpl.path.Path(c)
        mask = path.contains_points(points, radius=0)
        mask.shape = xv.shape

        # wtf?! path.contains_points returns everything head-over!
        mask = np.swapaxes(mask.reshape(mask.shape[0], mask.shape[1]), 0, 1)

        polygon_coords = np.argwhere(mask == True)
        # mask for RGB image
        final_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        # mask for binary mask
        final_bin_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for pixel in polygon_coords:
            # replace by RGB values
            final_mask[pixel[0], pixel[1]] = zero_mask[left[0] + pixel[0], left[1] + pixel[1], :]
            # replace with binary pattern
            final_bin_mask[pixel[0], pixel[1]] = orig_mask[left[0] + pixel[0], left[1] + pixel[1]]

        # ===================================================================================================================

        patch = final_mask

        # convert pixels to grayscale
        graypatch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)

        # # Plot result
        # fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
        # # Show RGB and segmentation mask
        # axs.imshow(patch)
        # axs.set_title('original patch')
        # plt.show(block=True)

        # calculate the grey level covariance matrix and extract Haralick texture features
        glcm = greycomatrix(graypatch, distances=[1, 2, 3, 4, 6, 8, 12], angles=[0, np.pi / 2], levels=256,
                            symmetric=True, normed=True)

        # zero intensity value observed for background pixels, obtain the GLCM of the ROI by simply discarding the first line
        # and the first column of the full image's GLCM.
        glcm_br = glcm[1:, 1:, :, :]

        try:
            glcm_br_norm = np.true_divide(glcm_br, glcm_br.sum(axis=(0, 1)))
            x = greycoprops(glcm_br_norm, prop='dissimilarity')
            x = x.flatten()
        except:
            x = np.empty((14,))
            x[:] = np.NaN

        gcprops = []
        for prop in range(len(x)):
            d = {f'greycoprop_{prop + 1}': x[prop]}
            gcprops.append(d)

        greycopreds = {}
        for prop in gcprops:
            greycopreds.update(prop)

        # get contours of original components
        mask_component = morphology.area_closing(final_bin_mask, area_threshold=50)
        mask_component_erode = morphology.erosion(mask_component, selem=np.ones((2, 2), np.uint8))
        coords = np.where(mask_component_erode == 0)

        # calculate entropy
        img_ent = Entropy(graypatch, morphology.disk(2))

        # remove border region
        img_ent_def = np.where(mask_component_erode == 255, img_ent, 0)

        entr_value = []
        for pixel in polygon_coords:
            entr_value.append(img_ent_def[pixel[0]][pixel[1]])
        entr_value = [i for i in entr_value if i != 0]

        entr_mean = np.mean(entr_value)
        entr_median = np.median(entr_value)
        entr_kurt = kurtosis(entr_value)
        entr_skew = skew(entr_value)
        entr_entr = entropy(entr_value)

        # # Plot distribution
        # plt.hist(entr_value, bins=50)
        # plt.show()

        # # Plot result
        # fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
        # # Show RGB and segmentation mask
        # axs[0].imshow(patch)
        # axs[0].set_title('original patch')
        # axs[1].imshow(img_ent)
        # axs[1].set_title('entropx')
        # axs[2].imshow(img_ent_def)
        # axs[2].set_title('entropy_erosion')
        # plt.show(block=True)

        # ======================================================================================================================

        # COLOR PROPERTIES

        color_spaces, descriptors, descriptor_names = ImageFunctions.get_color_spaces(img)

        # calculate descriptive statistics of color descriptors
        col_preds = []
        for descriptor in range(0, descriptors.shape[2]):
            desc_name = descriptor_names[descriptor]
            desc = descriptors[:, :, descriptor]
            orig_mask = np.zeros(img.shape[:2], dtype=np.float32)
            value = []
            for p in polygon:
                orig_mask[p[0], p[1]] = desc[p[0], p[1]]
                value.append(desc[p[0], p[1]])

            # mask for values
            final_bin_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.float32)
            for pixel in polygon_coords:
                # replace with binary pattern
                final_bin_mask[pixel[0], pixel[1]] = orig_mask[left[0] + pixel[0], left[1] + pixel[1]]

            col_value = []
            for pixel in polygon:
                col_value.append(desc[pixel[0]][pixel[1]])

            mn = np.mean(col_value)
            md = np.median(col_value)
            kt = kurtosis(col_value)
            sk = skew(col_value)
            en = entropy(col_value)

            d = {f'{desc_name}_mean': mn, f'{desc_name}_median': md, f'{desc_name}_kurt': kt,
                 f'{desc_name}_skew': sk, f'{desc_name}_entr': en}

            col_preds.append(d)

        colorpreds = {}
        for col in col_preds:
            colorpreds.update(col)

        XY = ({'class_label': labs[i],
               'ctype': ctype,
               'min_dist_to_row_ctr': min_dist_to_row_ctr, 'tgi_ctr': tgi_ctr,
               'share_veg': share_veg, 'n_segments': n_segments, 'max_ang_soil': max_ang_soil,
               'prop_longest': prop_longest,
               'n_holes': n_holes, 'area_ratio_holes': area_ratio_holes,
               'area': area,
               'elong': elong, 'elong_m': elong_m,
               'orientation': orientation,
               'eccentricity': eccentricity,
               'n_convdefs': n_convdefs, 'area_ratio': area_ratio,
               'roundness': roundness,
               'entr_mean': entr_mean, 'entr_median': entr_median, 'entr_kurt': entr_kurt,
               'entr_skew': entr_skew, 'entr_entr': entr_entr})

        XY = {**XY, **greycopreds, **colorpreds}
        preds.append(XY)

    return preds


def get_object_descriptors_hh_pred(img, orig_mask_filtered, pp_mask, rows, idx_map, reconstruct):
    # get color descriptors for image
    color_spaces, descriptors, descriptor_names = ImageFunctions.get_color_spaces(img)

    # # reconstruct objects using watershed segmentation if required
    # if reconstruct:
    #     pp_mask = reconstruct_eroded_objects(pp_mask, orig_mask_filtered)

    # get components
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(pp_mask, connectivity=8)

    # component label index must be inverted, because cv2.findContours and cv2.connectedComponentsWithStats
    # start at opposite ends of the image... awesome!
    outputr = n - labels
    outputr = np.where(outputr == n, 0, outputr - 1)

    # get contours components
    ccc, _ = cv2.findContours(pp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # check
    if not (n - 1 == len(ccc)):
        sys.exit("something wrong with the number of components!")

    # get contour type information for the new contour ordering
    preds = []
    for i, c in enumerate(ccc):

        # get the contour type based on the pixel values lying within the contour
        # use max() in case there are wholes in the component
        # component label index must be inverted, because cv2.findContours and cv2.connectedComponentsWithStats
        # start at opposite ends of the image... awesome!
        cnt_type = max(np.unique(pp_mask[np.where(labels == n - 1 - i)]))
        if cnt_type == 255:
            ctype = "eroded"
            dfact = list(range(18, 30))
        elif cnt_type == 125:
            ctype = "original"
            dfact = list(range(2, 14))

        # iterate over different levels of dilation
        vector1 = []
        vector2 = []
        blank0 = np.zeros(img.shape[:2], dtype=np.uint8)
        img2 = copy.copy(img)
        for k in range(len(dfact)):
            # dilate contour
            cont, hull = dilate_contour(img=img, contour=c, ctype=cnt_type, npix=dfact[k])

            # # visual output
            # img2 = cv2.drawContours(img2, cont, -1, (0, 0, 255), 1)
            # blank0 = cv2.drawContours(blank0, cont, -1, 125, 1)

            # get contour centroid
            ctr_hull = get_centroid(contour=hull)
            ctr_cont = get_centroid(contour=cont[0])
            # get coordinates of contour points and hull points
            c_cont = get_cont_coordinates(contour=cont[0], img=img)
            c_hull = get_cont_coordinates(contour=hull, img=img)
            # transform from cartesian to polar coordinates
            polar_coords_cont = utils.cart2pol(x=c_cont[1], y=c_cont[0], ctr=ctr_cont)
            polar_coords_hull = utils.cart2pol(x=c_hull[1], y=c_hull[0], ctr=ctr_hull)

            # get pixel values on hull
            v1 = scan_format_contour(coords=c_hull, polar_coords=polar_coords_hull,
                                     orig_mask_filtered=orig_mask_filtered)
            # get pixel values contour
            v2 = scan_format_contour(coords=c_cont, polar_coords=polar_coords_cont,
                                     orig_mask_filtered=orig_mask_filtered)
            vector1.append(v1)
            vector2.append(v2)

        # summarise contour scans
        max_ang_soil = summarise_scan(result=vector1, type="hull", ctype=ctype,
                                      regrown=reconstruct)  # get the maximum angle from convex hull
        share_veg, n_segments, prop_longest = summarise_scan(result=vector2, type="cont", ctype=ctype,
                                                             regrown=reconstruct)  # ratio and the number of disconnected segments from contour

        blank = np.zeros(img.shape[:2], dtype=np.uint8)
        blank1 = copy.copy(blank)
        # for the large contours, dilation has already been done; draw contour onto blank
        if cnt_type == 255:
            cv2.drawContours(blank, [c], -1, 255, 36)
        elif cnt_type == 125:
            cv2.drawContours(blank, [c], -1, 255, 18)
        cv2.fillPoly(blank, pts=c, color=255)

        # # Plot result
        # fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
        # # Show RGB and segmentation mask
        # axs[0].imshow(blank)
        # axs[0].set_title('contour')
        # axs[1].imshow(blank1)
        # axs[1].set_title('mask')
        # plt.show(block=True)

        # this is needed for contours lying on the image edge!!
        res0 = cv2.medianBlur(blank, 5)
        # get the dilated contour
        cnt_dil, _ = cv2.findContours(res0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blank1, cnt_dil, -1, 125, 1)

        xs = []
        ys = []
        for point in cnt_dil[0]:
            x = point[0][1]
            y = point[0][0]
            xs.append(x)
            ys.append(y)
        point_list = []
        for a, b in zip(xs, ys):
            point_list.append([a, b])
        c = point_list

        # get pixel coordinates and pixel values of the contour
        coords = np.where(blank1 == 125)
        cnt_vals = orig_mask_filtered[coords]
        # get veg/soil ratio of contour pixels
        npix_soil = len(np.where(cnt_vals == 0)[0])
        npix_veg = len(np.where(cnt_vals == 255)[0])
        ratio_veg_soil = npix_veg / (npix_veg + npix_soil)

        # # Plot result
        # fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
        # # Show RGB and segmentation mask
        # axs[0].imshow(blank1)
        # axs[0].set_title('contour')
        # axs[1].imshow(mask_filtered)
        # axs[1].set_title('mask')
        # plt.show(block=True)

        print(f'{i}/{n}')
        polygon = np.array(c)

        # get the bounding box
        left = np.min(polygon, axis=0)
        right = np.max(polygon, axis=0)
        x = np.arange(math.ceil(left[0]), math.floor(right[0]) + 1)
        y = np.arange(math.ceil(left[1]), math.floor(right[1]) + 1)
        xv, yv = np.meshgrid(x, y, indexing='xy')
        points = np.hstack((xv.reshape((-1, 1)), yv.reshape((-1, 1))))

        # define empty masks (2D and 3D)
        zero_mask = np.zeros(img.shape, dtype=np.uint8)
        orig_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        # fill bounding box
        # this contains all pixel values of the bounding box: needs further processing
        for p in points:
            zero_mask[p[0], p[1]] = img[p[0], p[1], :]
            orig_mask[p[0], p[1]] = orig_mask_filtered[p[0], p[1]]

        # # Plot result
        # fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
        # # Show RGB and segmentation mask
        # axs[0].imshow(zero_mask)
        # axs[0].set_title('contour')
        # axs[1].imshow(orig_mask)
        # axs[1].set_title('mask')
        # axs[2].imshow(orig_mask_filtered)
        # axs[2].set_title('mask')
        # plt.show(block=True)

        # ==================================================================================================================

        # detect components and keep only the main component, removing smaller ones, neighbouring ones, and noise
        # get centroid coordinates for later use
        n_comps, output, stats, centroids = cv2.connectedComponentsWithStats(orig_mask, connectivity=8)
        sizes = list(stats[:, 4][1:])
        index = sizes.index(np.max(sizes))
        bin = np.where(output == index + 1, True, False)
        bin_num = np.uint8(np.where(output == index + 1, 1, 0))
        ctr = centroids[index + 1]
        # done

        # ==================================================================================================================

        # EXTRACT FEATURES FOR THE REMAINING COMPONENT
        # POSITION and SHAPE

        # features related to centroid position
        rowpoints = np.transpose(np.nonzero(rows))
        min_dist_to_row_ctr = np.amin(cdist([ctr], rowpoints, 'euclidean'))
        tgi_ctr = idx_map[int(ctr[1]), int(ctr[0])]

        # features to use from skimage.measure.regionprops
        label_img = measure.label(bin_num, connectivity=2)
        props = measure.regionprops(label_img)
        orientation = props[0].orientation
        eccentricity = props[0].eccentricity

        # custom features
        # get the component contour
        contour, _ = cv2.findContours(bin_num, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # measures of object elongation
        rect = cv2.minAreaRect(contour[0])
        bbox = np.int0(cv2.boxPoints(rect))
        (tl, tr, br, bl) = bbox
        xdist1 = tl[0] - tr[0]
        ydist1 = tl[1] - tr[1]
        length1 = math.sqrt(xdist1 * xdist1 + ydist1 * ydist1)
        xdist2 = tl[0] - bl[0]
        ydist2 = tl[1] - bl[1]
        length2 = math.sqrt(xdist2 * xdist2 + ydist2 * ydist2)
        long = max(length1, length2)
        short = min(length1, length2)
        elong = short / long

        # alternative object elongation measure
        def elongation(m):
            x = m['mu20'] + m['mu02']
            y = 4 * m['mu11'] ** 2 + (m['mu20'] - m['mu02']) ** 2
            return (x + y ** 0.5) / (x - y ** 0.5)

        m = cv2.moments(bin_num)
        elong_m = elongation(m)

        # area ratio (component/convex hull)
        area = sizes[index]  # area of component
        points = utils.flatten_contour_data(contour, asarray=True)  # reshape point data
        hull = ConvexHull(points)
        area_hull = hull.volume
        area_ratio = area / area_hull

        # roundness
        vertices = hull.vertices.tolist() + [hull.vertices[0]]
        perimeter = np.sum([distance.euclidean(x, y) for x, y in zip(points[vertices], points[vertices][1:])])
        roundness = (4 * math.pi * area) / (perimeter * perimeter)

        # convexity defects
        hull = cv2.convexHull(contour[0], returnPoints=False)
        defects = cv2.convexityDefects(contour[0], hull)
        if defects is not None:
            n_convdefs = len(defects)
        else:
            n_convdefs = 0

        # ==================================================================================================================

        # coordinates of pixels making up the plant object
        x, y = np.where(output == index + 1)
        coords = []
        for a, b in zip(x, y):
            coords.append([a, b])
        polygon = np.array(coords)

        # new bounding box for the kept component
        left = np.min(polygon, axis=0)
        right = np.max(polygon, axis=0)
        x = np.arange(math.ceil(left[0]), math.floor(right[0]) + 1)
        y = np.arange(math.ceil(left[1]), math.floor(right[1]) + 1)
        xv, yv = np.meshgrid(x, y, indexing='xy')
        points = np.hstack((xv.reshape((-1, 1)), yv.reshape((-1, 1))))

        # define empty masks (2D and 3D)
        zero_mask = np.zeros(img.shape, dtype=np.uint8)
        orig_mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # fill for the coordinates containing plant
        for p in polygon:
            zero_mask[p[0], p[1]] = img[p[0], p[1], :]
            orig_mask[p[0], p[1]] = orig_mask_filtered[p[0], p[1]]

        # # Plot result
        # fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
        # # Show RGB and segmentation mask
        # axs[0].imshow(zero_mask)
        # axs[0].set_title('contour')
        # axs[1].imshow(orig_mask)
        # axs[1].set_title('mask')
        # axs[2].imshow(orig_mask_filtered)
        # axs[2].set_title('mask')
        # plt.show(block=True)

        # ===================================================================================================================

        # identify pixels lying within the contour
        # replace with actual RGB values or with binary vegetation/background value
        path = mpl.path.Path(c)
        mask = path.contains_points(points, radius=0)
        mask.shape = xv.shape

        # wtf?! path.contains_points returns everything head-over!
        mask = np.swapaxes(mask.reshape(mask.shape[0], mask.shape[1]), 0, 1)

        polygon_coords = np.argwhere(mask == True)
        # mask for RGB image
        final_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        # mask for binary mask
        final_bin_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for pixel in polygon_coords:
            # replace by RGB values
            final_mask[pixel[0], pixel[1]] = zero_mask[left[0] + pixel[0], left[1] + pixel[1], :]
            # replace with binary pattern
            final_bin_mask[pixel[0], pixel[1]] = orig_mask[left[0] + pixel[0], left[1] + pixel[1]]

        # ===================================================================================================================

        patch = final_mask

        # convert pixels to grayscale
        graypatch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)

        # # Plot result
        # fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
        # # Show RGB and segmentation mask
        # axs.imshow(patch)
        # axs.set_title('original patch')
        # plt.show(block=True)

        # calculate the grey level covariance matrix and extract Haralick texture features
        glcm = greycomatrix(graypatch, distances=[1, 2, 3, 4, 6, 8, 12], angles=[0, np.pi / 2], levels=256,
                            symmetric=True, normed=True)

        # zero intensity value observed for background pixels, obtain the GLCM of the ROI by simply discarding the first line
        # and the first column of the full image's GLCM.
        glcm_br = glcm[1:, 1:, :, :]

        try:
            glcm_br_norm = np.true_divide(glcm_br, glcm_br.sum(axis=(0, 1)))
            x = greycoprops(glcm_br_norm, prop='dissimilarity')
            x = x.flatten()
        except:
            x = np.empty((14,))
            x[:] = np.NaN

        gcprops = []
        for prop in range(len(x)):
            d = {f'greycoprop_{prop + 1}': x[prop]}
            gcprops.append(d)

        greycopreds = {}
        for prop in gcprops:
            greycopreds.update(prop)

        # get contours of original components
        mask_component = morphology.area_closing(final_bin_mask, area_threshold=50)
        mask_component_erode = morphology.erosion(mask_component, selem=np.ones((2, 2), np.uint8))

        # calculate entropy
        img_ent = Entropy(graypatch, morphology.disk(2))

        # remove border region
        img_ent_def = np.where(mask_component_erode == 255, img_ent, 0)

        entr_value = []
        for pixel in polygon_coords:
            entr_value.append(img_ent_def[pixel[0]][pixel[1]])
        entr_value = [i for i in entr_value if i != 0]

        entr_mean = np.mean(entr_value)
        entr_median = np.median(entr_value)
        entr_kurt = kurtosis(entr_value)
        entr_skew = skew(entr_value)
        entr_entr = entropy(entr_value)

        # # Plot distribution
        # plt.hist(entr_value, bins=50)
        # plt.show()

        # # Plot result
        # fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
        # # Show RGB and segmentation mask
        # axs[0].imshow(patch)
        # axs[0].set_title('original patch')
        # axs[1].imshow(img_ent)
        # axs[1].set_title('entropx')
        # axs[2].imshow(img_ent_def)
        # axs[2].set_title('entropy_erosion')
        # plt.show(block=True)

        # ======================================================================================================================

        # COLOR PROPERTIES
        # calculate descriptive statistics of color descriptors
        col_preds = []
        for descriptor in range(0, descriptors.shape[2]):
            desc_name = descriptor_names[descriptor]
            desc = descriptors[:, :, descriptor]
            orig_mask = np.zeros(img.shape[:2], dtype=np.float32)
            value = []
            for p in polygon:
                orig_mask[p[0], p[1]] = desc[p[0], p[1]]
                value.append(desc[p[0], p[1]])

            # mask for values
            final_bin_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.float32)
            for pixel in polygon_coords:
                # replace with binary pattern
                final_bin_mask[pixel[0], pixel[1]] = orig_mask[left[0] + pixel[0], left[1] + pixel[1]]

            col_value = []
            for pixel in polygon:
                col_value.append(desc[pixel[0]][pixel[1]])

            mn = np.mean(col_value)
            md = np.median(col_value)
            kt = kurtosis(col_value)
            sk = skew(col_value)
            en = entropy(col_value)

            d = {f'{desc_name}_mean': mn, f'{desc_name}_median': md, f'{desc_name}_kurt': kt,
                 f'{desc_name}_skew': sk, f'{desc_name}_entr': en}

            col_preds.append(d)

        colorpreds = {}
        for col in col_preds:
            colorpreds.update(col)

        XY = ({'ctype': ctype,
               'min_dist_to_row_ctr': min_dist_to_row_ctr, 'tgi_ctr': tgi_ctr,
               'share_veg': share_veg, 'n_segments': n_segments, 'max_ang_soil': max_ang_soil,
               'prop_longest': prop_longest,
               'elong': elong, 'elong_m': elong_m,
               'orientation': orientation,
               'eccentricity': eccentricity,
               'n_convdefs': n_convdefs, 'area_ratio': area_ratio,
               'roundness': roundness,
               'entr_mean': entr_mean, 'entr_median': entr_median, 'entr_kurt': entr_kurt,
               'entr_skew': entr_skew, 'entr_entr': entr_entr})

        XY = {**XY, **greycopreds, **colorpreds}
        preds.append(XY)

    return outputr, preds


# Function to extract all predictors required by the random forest classifier
def get_object_descriptors_pred(img, mask_pp, rows, rowsprof):
    """ detect objects of a given input --> updated function of Jonas
        :param a_segmented: a segmented picture (after random forrest and stitching the tiles together again in our case)
        :param picture: the rgb picture of the segmented
        :param kernel_size: determine the size to smooth the segmented --> get rid of blur and artefacts of bad random forrest calssification
        --> the bigger the size the smoother the image
        :return: list of the contours of all detected object, the hierarchy of this objects, and image and and image with the contours shown on the image
        """

    mask3 = copy.copy(mask_pp)

    # get components
    n_comps, output, stats, centroids = cv2.connectedComponentsWithStats(mask3, connectivity=8)

    # get contours
    # HERE, ONLY CONTOURS OF HIERARCHY 1 CAN BE USED (I.E. RETR_EXTERNAL),
    # AS THE NUMBER OF COMPONENTS MUST EQUAL THE NUMBER OF CONTOURS!!
    cnts, hier = cv2.findContours(mask3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # remove background
    areas = stats[1:, -1];
    centroids = centroids[1:, ]
    n_comps = n_comps - 1

    # ==================================================================================================================

    # for each component, get area
    area = []
    for i in range(0, n_comps):
        area.append(areas[i])

    # for each detected component contour, get
    # i)    maximum depth of the convexity defects;
    # ii)   eccentricity;
    # iii)  compactness
    max_depth_convdef = []
    for cont in range(0, len(cnts)):
        hull = cv2.convexHull(cnts[cont], returnPoints=False)
        defects = cv2.convexityDefects(cnts[cont], hull)
        # if there are any convexity defects
        if not defects is None:
            depth_convdef = []
            for defect in defects:
                depth_convdef.append(defect[:, 3][0])
            max_depth_convdef.append(max(depth_convdef))
        # if no convexity defects are detected
        else:
            max_depth_convdef.append(0)

    eccentricity = []
    compactness = []
    for cnt in cnts:
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            # center, axis_length and orientation of ellipse
            (center, axes, orientation) = ellipse
            # length of major and minor axis
            majoraxis_length = max(axes)
            minoraxis_length = min(axes)
            ecc = np.sqrt(1 - (minoraxis_length / majoraxis_length) ** 2)
            # compactness
            A = cv2.contourArea(cnt)
            equi_diameter = np.sqrt(4 * A / np.pi)
            comp = equi_diameter / majoraxis_length
        else:
            ecc = np.NaN
            comp = 0.0
        eccentricity.append(ecc)
        compactness.append(comp)

    # ==================================================================================================================

    color_spaces, descriptors, descriptor_names = ImageFunctions.demosaic_8bit_image(img)
    descriptors = descriptors[:, :, 3:14]

    def average_preds(tpl):
        out = []
        for x in range(0, 11):
            res = [j[x] for j in tpl]
            res = statistics.mean(res)
            out.append(res)
        return out

    colorpreds = []
    for i in range(1, n_comps + 1):
        pix_idx = np.where(output == i)
        dd = descriptors[pix_idx]
        R, G, B, H, S, V, L, a, b, ExG, ExR = average_preds(dd)
        col = ({'R': R, 'G': G, 'B': B,
                'H': H, 'S': S, 'V': V,
                'L': L, 'a': a, 'b': b,
                'ExG': ExG, 'ExR': ExR})
        colorpreds.append(col)

    # positional information
    rowpoints = np.transpose(np.nonzero(rows))
    min_dist_to_row = []
    tgi_ctr = []
    for i in range(0, len(centroids)):
        c = centroids[i]
        dist = np.amin(cdist(np.array([c]), rowpoints, 'euclidean'))
        tgi = rowsprof[int(c[1]), int(c[0])]
        min_dist_to_row.append(dist)
        tgi_ctr.append(tgi)

    # ==================================================================================================================

    # check output
    if not (len(centroids) == len(area) == len(max_depth_convdef) == len(eccentricity) == len(compactness) == len(min_dist_to_row) == len(tgi_ctr)):
        sys.exit("unequal ncomp and ncont")

    else:
        preds = []
        for i in range(0, len(centroids)):
            XY = ({'area': area[i], 'max_depth_convdef': max_depth_convdef[i],
                   'eccentricity': eccentricity[i], 'compactness': compactness[i],
                   'min_dist_to_row': min_dist_to_row[i], 'tgi_ctr': tgi_ctr[i]})
            XY = {**XY, **colorpreds[i]}
            preds.append(XY)
        return output, preds

# ======================================================================================================================
