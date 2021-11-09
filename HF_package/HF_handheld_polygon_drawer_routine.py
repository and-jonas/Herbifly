# Author: Flavian Tschurr
# Project: Herbifly
# Date: 23.06.2020
# Last edited: Jonas Anderegg, 2021-11-09

########################################################################################################################
# imports
########################################################################################################################

from pathlib import Path
import os

native_os_path_join = os.path.join


def modified_join(*args, **kwargs):
    return native_os_path_join(*args, **kwargs).replace('\\', '/')


os.path.join = modified_join

import matplotlib.image as mpimg
import HF_package.utils as utils
import HF_package.HandheldFunctions as HandheldFunctions
import HF_package.ImageFunctions as ImageFunctions

########################################################################################################################
# variables
########################################################################################################################

workdir = "O:/Evaluation/Hiwi/2020_Herbifly/Images_Farmers"
workdir_out = "O:/Evaluation/Hiwi//2020_Herbifly/Images_Farmers"
picture_type = "Handheld"
gridSize = 1
pic_format = ".JPG"
farmers = ["Egli", "Scheidegger", "Keller", "Bolli", "Bonny", "Miauton", "Baumberger2", "Baumberger1", "Stettler"]

########################################################################################################################
# start of the calculation and iteration over the farmer etc.
########################################################################################################################

for farmer in farmers:

    # create paths
    farmer_region = utils.get_farmer_region(farmer)
    path_myfarm = os.path.join(workdir, farmer_region, farmer, picture_type)
    path_grid_output =os.path.join(workdir_out, "Meta", farmer, picture_type)
    path_grid_output_grids=os.path.join(path_grid_output, "frames")
    dates = os.listdir(path_myfarm)

    for date in dates:
        print(date)
        if utils._check_date_name(date):
            path_myDate = os.path.join(path_myfarm, date)
            images = os.listdir(path_myDate)
            for image in images:
                print(image)
                if utils._check_image_name(image, pic_format):
                    pic_name = image[0:-len(pic_format)]

                    path_currentImage = os.path.join(path_myDate, image)

                    pictureCurrent_all = mpimg.imread(str(path_currentImage))

                    path_current_json = os.path.join(workdir, "Meta", farmer, picture_type, "frames", date)
                    Path(path_current_json).mkdir(parents=True, exist_ok=True)
                    if not Path("{path_current_j}/{pic_n}.geojson".format(path_current_j=path_current_json,
                                                                          pic_n=pic_name)).exists():
                        corners = ImageFunctions.capture_plot_shape_GUI(pictureCurrent_all)
                        HandheldFunctions.write_geojson_polygon_mask_handheld(corners=corners, image_name=pic_name,
                                                                              path_folder=path_current_json)


