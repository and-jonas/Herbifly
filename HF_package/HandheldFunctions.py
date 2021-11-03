### Author: Flavian Tschurr
### Project: Herbifly
### Script use: functions to detect and classify objects within a segmented piture
### Date: 06.05.2020
########################################################################################################################
# imports
########################################################################################################################
from matplotlib import path
import numpy as np
import geojson


def write_geojson_polygon_mask_handheld(corners, image_name, path_folder):
    '''Writes a geojson mask file containing one single polygon --> adapted from Lukas

    :param corners: Corners of polygon as list, first and last entry should be equal to recieve a closed polygon
    :param image_name: Name of the image the mask belongs to, no suffix (e.g. no '.JPG')
    :param path_folder: Path to the folder to store the file
    '''

    if corners:
        polygon = geojson.Polygon([corners])
        feature = geojson.Feature(geometry=polygon,
                                  properties={'image': image_name, 'type': 'handheld_mask'})
        feature_collection = geojson.FeatureCollection(features=[feature])
        # with open(f'{path_folder}/{image_name}.geojson', "w") as outfile:
        with open("{folder}/{img_n}.geojson".format(folder=path_folder, img_n=image_name), 'w') as outfile:
            geojson.dump(feature_collection, outfile, indent='\t')
    else:
        print('Empty corners list, nothing to write')


def handheld_coverage_calculator(polygon_mask, contour_mask):
    """
    :param polygon_mask:
    :param contour_mask:
    :return:
    """
    polygon = polygon_mask[0]["geometry"]["coordinates"][0]
    # transform coordinates to a path
    polygon_path = path.Path(polygon, closed=True)

    # create a mask of the image
    xcoords = np.arange(0, contour_mask.shape[1])
    ycoords = np.arange(0, contour_mask.shape[0])
    coords = np.transpose([np.repeat(ycoords, len(xcoords)), np.tile(xcoords, len(ycoords))])

    # mask polygon
    polygon_mask_image = polygon_path.contains_points(coords, radius=1)
    polygon_mask_image = np.swapaxes(polygon_mask_image.reshape(contour_mask.shape[0], contour_mask.shape[1]), 0, 1)
    polygon_coords = np.argwhere(polygon_mask_image)

    # count non-zero pixels (weed pixels)
    weed_counter = 0
    for pixel in polygon_coords:
        if contour_mask[pixel[1]][pixel[0]] != 0:
            weed_counter = weed_counter + 1

    # calculate the ratio
    if len(polygon_coords) > 0:
        cover_ratio = weed_counter / len(polygon_coords)
    else:
        cover_ratio = None

    output = [cover_ratio, len(polygon_coords), weed_counter]
    return output


def frame_length_reader(polygon_mask):
    """
    Calculation of the length of the frame in the field (of the handheld fotos out of a geojson)
    :param polygon_mask: geojson with the pixeöl coordinates of the corners of a frame
    :return: mean of the side length(approx. 1m) in pixels --> as a multiplier for the object detection
    """
    polygon = polygon_mask[0]["geometry"]["coordinates"][0]
    length_list=[]
    for corner in range(0,len(polygon)-1):
        x_dir = (polygon[corner][0]-polygon[corner+1][0])**2
        y_dir = (polygon[corner][1]-polygon[corner+1][1])**2
        length_list.append((x_dir+y_dir)**(1/2))
    reference_meter = sum(length_list)/len(length_list)
    return reference_meter


def handheld_coverage_calculator_index(polygon_mask,contour_mask):
    """

    :param polygon_mask:
    :param contour_mask:
    :return:
    """
    polygon = polygon_mask[0]["geometry"]["coordinates"][0]
    # transofrm the coordinates to a path
    polygon_path = path.Path(polygon,closed = False)

    # create a mask of the image
    xcoords = np.arange(0, contour_mask.shape[1])
    ycoords = np.arange(0, contour_mask.shape[0])
    # coords = np.transpose([np.repeat(xcoords, len(ycoords)), np.tile(ycoords, len(xcoords))])
    coords = np.transpose([np.repeat(ycoords, len(xcoords)), np.tile(xcoords, len(ycoords))])

    # maskin of the polygon
    polygon_mask_image = polygon_path.contains_points(coords, radius=0)

    polygon_mask_image = np.swapaxes(polygon_mask_image.reshape(contour_mask.shape[0], contour_mask.shape[1]), 0, 1)
    # polygon_mask_image = np.swapaxes(polygon_mask_image.reshape(contour_mask.shape[1], contour_mask.shape[0]), 0, 1)

    polygon_coords = np.argwhere(polygon_mask_image == True)
    # we count all pixels which are filled in the contour_mask --> the non 0 values are weeds as we sorted the rest out earlier
    mean_out = None
    if len(polygon_coords > 0):
        Index_value = []
        for pixel in polygon_coords:
            # GLI_value.append(contour_mask[pixel[1]][pixel[0]])
            Index_value.append(contour_mask[pixel[0]][pixel[1]])

        mean_out = np.mean(Index_value)
    else:
        # cover_ratio = 99999
        mean_out = None
    return mean_out



