
# Author: Flavian Tschurr
# Project: Herbifly; this Module contains functions used for the frame validation wrapper script
# Date: 17.06.2020

########################################################################################################################
# imports
import numpy as np
from matplotlib import path
########################################################################################################################


def image_finder(cornersDF, polygon_coords):

    poly_x = []
    poly_y = []
    img_of_interest= []
    for i in range(0,len(polygon_coords)):
        poly_x.append(polygon_coords[i][0])
        poly_y.append(polygon_coords[i][1])

    # extreme values of the frame
    poly_x_max = float(max(poly_x))
    poly_x_min = float(min(poly_x))
    poly_y_max = float(max(poly_y))
    poly_y_min = float(min(poly_y))
    # poly_x_max = 2604895.9
    # poly_x_min = 2604895.7
    # poly_y_max = 1225104.8
    # poly_y_min = 1225104.6

    for image in range(0,len(cornersDF.camera)):
        # print(image)
        img_x=[float(cornersDF.loc[image:image].e1_x),float(cornersDF.loc[image:image].e2_x),
               float(cornersDF.loc[image:image].e3_x),float(cornersDF.loc[image:image].e4_x)]
        img_y=[float(cornersDF.loc[image:image].e1_y),float(cornersDF.loc[image:image].e2_y),
               float(cornersDF.loc[image:image].e3_y),float(cornersDF.loc[image:image].e4_y)]

        if min(img_x) < poly_x_min < max(img_x) and min(img_x) < poly_x_max < max(img_x) and min(img_y) < poly_y_min < max(img_y) and min(img_y) < poly_y_max < max(img_y):
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


