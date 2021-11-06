
# ======================================================================================================================
# Author: Flavian Tschurr, Jonas Anderegg
# Project: Herbifly; this Module contains functions used for the frame validation wrapper script
# Date: 17.06.2020
# Last modified: Jonas Anderegg, 2021-11-03
# ======================================================================================================================

# imports
import numpy as np
from matplotlib import path
import os
import geojson
import HF_package.utils as utils

# ======================================================================================================================


def image_finder(cornersDF, polygon_coords):

    poly_x = []
    poly_y = []
    img_of_interest = []
    for i in range(0, len(polygon_coords)):
        poly_x.append(polygon_coords[i][0])
        poly_y.append(polygon_coords[i][1])

    # extreme values of the frame
    poly_x_max = float(max(poly_x))
    poly_x_min = float(min(poly_x))
    poly_y_max = float(max(poly_y))
    poly_y_min = float(min(poly_y))

    for image in range(0, len(cornersDF.camera)):
        # print(image)
        img_x = [float(cornersDF.loc[image:image].e1_x), float(cornersDF.loc[image:image].e2_x),
                 float(cornersDF.loc[image:image].e3_x), float(cornersDF.loc[image:image].e4_x)]
        img_y = [float(cornersDF.loc[image:image].e1_y), float(cornersDF.loc[image:image].e2_y),
                 float(cornersDF.loc[image:image].e3_y), float(cornersDF.loc[image:image].e4_y)]

        # if min(img_x) < poly_x_min < max(img_x) and min(img_x) < poly_x_max < max(img_x) and min(img_y) < poly_y_min < max(img_y) and min(img_y) < poly_y_max < max(img_y):
        #     img_of_interest.append([cornersDF.loc[image].camera, image])

        # select more conservatively:
        if min(img_x)+2 < poly_x_min < max(img_x)-2 and min(img_x)+2 < poly_x_max < max(img_x)-2 and \
                min(img_y)+2 < poly_y_min < max(img_y)-2 and min(img_y)+2 < poly_y_max < max(img_y)-2:
            img_of_interest.append([cornersDF.loc[image].camera, image])

    return img_of_interest


def polygon_value_calculator(corner_grid_pic, contour_mask):
    """

    :param corner_grid_pic: corners of a grid within a picture
    :param contour_mask: detected contours
    :return: coverage as a ratio
    """

    # transform coordinates to a path
    grid_path = path.Path(corner_grid_pic,closed=False)

    # create a mask of the image
    xcoords = np.arange(0, contour_mask.shape[0])
    ycoords = np.arange(0, contour_mask.shape[1])
    coords = np.transpose([np.repeat(ycoords, len(xcoords)), np.tile(xcoords, len(ycoords))])

    # Create mask
    grid_mask_image = grid_path.contains_points(coords, radius=-0.5)
    ## check if correct or if the axes needed to be swaped! --> change corresponding line in code
    grid_mask_image = np.swapaxes(grid_mask_image.reshape(contour_mask.shape[1], contour_mask.shape[0]), 0, 1)
    # grid_mask_image = np.swapaxes(grid_mask_image.reshape(contour_mask.shape[0], contour_mask.shape[1]), 0, 1)
    # grid_mask_image = grid_mask_image.reshape(contour_mask.shape[0], contour_mask.shape[a])
    # we take the coordiantes of the polygon --> these are masked now
    polygon_coords = np.argwhere(grid_mask_image == True)

    # B = grid_mask_image.astype(int)

    # we count all pixels which are filled in the contour_mask --> the non 0 values are weeds as we sorted the rest out earlier
    mean_out= None
    if len(polygon_coords >0):
        GLI_value= []
        for pixel in polygon_coords:
            GLI_value.append(contour_mask[pixel[0]][pixel[1]])
        mean_out = np.mean(GLI_value)
    else:
        # cover_ratio = 99999
        mean_out = None
    return mean_out


def filter_images_frames(path_current_json, cornersDF):
    geojsons = os.listdir(path_current_json)
    for geoj in geojsons:
        if utils._check_geojson(geoj):
            path_geojson_current = os.path.join(path_current_json)  # needs to be adapted!
            with open("{path_geo}/{pic_n}".format(path_geo=path_geojson_current, pic_n=geoj),
                      'r') as infile:
                polygon_mask = geojson.load(infile)
                polygons = polygon_mask["features"]
                # iterate over the polygons within the geojson file
                images_use = []
                for polygon in polygons:
                    coords = polygon["geometry"]["coordinates"][0][0]  # might be needed for 10m?!
                    images = image_finder(cornersDF, coords)
                    try:
                        # images = images[0][0]  ## wtf ?????
                        images = [sel[0] for sel in images]
                    except IndexError:
                        continue
                    images_use.extend(images)
    return images_use