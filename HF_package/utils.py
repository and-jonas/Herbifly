
# ======================================================================================================================
# Author: Jonas Anderegg, Flavian Tschurr
# Project: Herbifly
# Script use: provides various utility functions for the HF_package
# Date: 06.05.2020
# Last edited: Jonas Anderegg,  2021-11-01
# ======================================================================================================================

# imports
import re
import cv2
import pandas as pd
import math
import numpy as np
from random import randint
import statistics


# function to check if the folder is in the correct format (leave out wrong folders)
def _check_date_name(date):
    """ check if the input is a dat in our format
    :param date: input string (name of a folder in our case)
    :return: True/ None
    """
    return re.match(r'\d{8}', date)


def _check_image_name(image, pic_format):
    """ check if the input is an image of our definition
    :param image: name of an image (without the format e.g. ".jpg"
    :param pic_format: format of the input e.g.: ".jpg"
    :return: True False
    """
    return re.search(pic_format, image, re.IGNORECASE)


def _check_geojson(geojson_file):
    """ check if a certain file name contains to a geojson file
    :param geojson_file: input filename
    :return: True/ False
    """
    format = ".geojson"
    return format in geojson_file


def weed_coverage(contourNumbers, contours, img_allContours):
    """
    :param contourNumbers: list of contours
    :param contours: copuntours itself
    :param img_allContours: img with contours on it
    :return:
    """

    pixelArea = []
    cols = ["pixel"]
    for nr in contourNumbers:
        pixelArea.append(len(contours[int(nr)]))
    pixelArea = pd.DataFrame(pixelArea, columns=cols)

    contours_filtered = []
    for nr in contourNumbers:
        contours_filtered.append(contours[int(nr)])

    image_weed = cv2.drawContours(img_allContours, contours_filtered, -1, (255, 0, 0), 2)

    return pixelArea, contours_filtered, image_weed


def round_up_to_even(f):
    """
    rounds a number up to the enxt even number
    :param f: number
    :return: even number
    """
    return math.ceil(f / 2.) * 2


def round_down_to_even(f):
    """
    rounds a number down to the next even number
    :param f: number
    :return: even number
    """
    return math.floor(f / 2.) * 2


def get_farmer_region(farmer):
    if farmer == "Baumberger2" or farmer == "Baumberger1" or farmer == "Stettler":
        farmer_region = "Bern_Solothurn"
    elif farmer == "Egli" or farmer == "Keller" or farmer == "Bolli":
        farmer_region = "Nordostschweiz"
    elif farmer == "Scheidegger" or farmer == "Miauton" or farmer == "Bonny":
        farmer_region = "Broye"
    elif farmer == "test":
        farmer_region = "test"
    return farmer_region


def cart2pol(x, y, ctr):
    x = x - ctr[1]
    y = y - ctr[0]
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    # phi = 180 * phi / math.pi
    return rho, phi


def pol2cart(rho, phi, ctr):
    x = rho * np.cos(phi) + ctr[1]
    y = rho * np.sin(phi) + ctr[0]
    return y.astype(int), x.astype(int)


def make_point_list(input):
    xs = []
    ys = []
    for point in range(len(input[0])):
        x = input[0][point]
        y = input[1][point]
        xs.append(x)
        ys.append(y)
    point_list = []
    for a, b in zip(xs, ys):
        point_list.append([a, b])
    c = point_list
    return c


def flatten_contour_data(input, asarray):
    xs = []
    ys = []
    for point in input[0]:
        x = point[0][1]
        y = point[0][0]
        xs.append(x)
        ys.append(y)
    point_list = []
    for a, b in zip(xs, ys):
        point_list.append([a, b])
        c = point_list
    if asarray:
        c = np.asarray(point_list)
    return c


def flatten_centroid_data(input, asarray):
    xs = []
    ys = []
    for point in input:
        x = point[1]
        y = point[0]
        xs.append(x)
        ys.append(y)
    point_list = []
    for a, b in zip(xs, ys):
        point_list.append([a, b])
        c = point_list
    if asarray:
        c = np.asarray(point_list)
    return c


def random_with_N_digits(n):
    range_start = 10**(n-1)
    range_end = (10**n)-1
    return randint(range_start, range_end)


def most_frequent(list):
    counter = 0
    num = list[0]
    for i in list:
        curr_frequency = list.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            num = i
    return num


def set_contour_dilation_factors(picture_type, contour_type, reconstruct):
    if picture_type == "Handheld":
        if reconstruct:
            if contour_type == 255:
                ctype = 'eroded'
                dfact = list(range(2, 14))
            elif contour_type == 125:
                ctype = 'original'
                dfact = list(range(2, 14))
        else:
            if contour_type == 255:
                ctype = 'eroded'
                dfact = list(range(18, 30))
            elif contour_type == 125:
                ctype = 'original'
                dfact = list(range(2, 14))
    elif picture_type == "10m":
        if contour_type == 125:
            ctype = 'original'
            dfact = list(range(2, 6))
    return ctype, dfact


def average_preds(tpl):
    out = []
    for x in range(0, 11):
        res = [j[x] for j in tpl]
        res = statistics.mean(res)
        out.append(res)
    return out


def filter_objects_size(mask, size_th, dir):
    """
    Filter objects in a binary mask by size
    :param mask: A binary mask to filter
    :param size_th: The size threshold used to filter (objects GREATER than the threshold will be kept)
    :return: A binary mask containing only objects greater than the specified threshold
    """
    _, output, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    sizes = stats[1:, -1]
    if dir == "greater":
        idx = (np.where(sizes > size_th)[0] + 1).tolist()
    if dir == "smaller":
        idx = (np.where(sizes < size_th)[0] + 1).tolist()
    out = np.in1d(output, idx).reshape(output.shape)
    cleaned = np.where(out, 125, mask*255)

    return cleaned
