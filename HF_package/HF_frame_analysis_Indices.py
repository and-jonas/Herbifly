
# ======================================================================================================================
# Author: Flavian Tschurr
# Project: Herbifly
# Script use: wrapper script to analyse frames (as geojson) from an orthomosaic on single images
# This wrapper uses the output masks of the HF_segmentation_main!! It is not a standalone script, no segmentation and
# classification is applied to reduce comnputing time and bias from different segmentation results
# Date: 17.06.2020
# Last edited: Jonas Anderegg, 2021-11-02
# ======================================================================================================================

# imports
import pandas as pd
import os

native_os_path_join = os.path.join
def modified_join(*args, **kwargs):
    return native_os_path_join(*args, **kwargs).replace('\\', '/')


os.path.join = modified_join


import geojson
import Metashape
from pathlib import Path
import matplotlib.image as mpimg
import copy

import HF_package.utils as utils
import HF_package.FrameFunctions as FrameFunctions
import HF_package.AgisoftFunctions as AgisoftFunctions
import HF_package.plant_indices as plant_indices

# ======================================================================================================================

# set variables
farmers = ["Baumberger1", "Baumberger2", "Stettler", "Egli", "Scheidegger", "Keller", "Bolli", "Bonny", "Miauton"]
farmers = ["Baumberger1"]
agisoft_path = "O:/Hiwi/2020_Herbifly/Processed_Campaigns"
workdir = "O:/Hiwi/2020_Herbifly/Images_Farmers"
picture_type = "30m"
picture_format = "JPG"

# indices to extract
index_names = ["TGI", "ExG", "ExGR", "GLI", "NDI", "VEG"]
index_names = ["TGI"]

# ======================================================================================================================

# iterate over farmers
for farmer in farmers:

    # iterate over indices
    for index in index_names:

        print(index)

        # fix index function
        if index == "TGI":
            index_function = plant_indices.index_TGI
        elif index == "ExG":
            index_function = plant_indices.index_ExG
        elif index == "ExGR":
            index_function = plant_indices.index_ExGR
        elif index == "GLI":
            index_function = plant_indices.index_GLI
        elif index == "NDI":
            index_function = plant_indices.index_NDI
        elif index == "VEG":
            index_function = plant_indices.index_VEG

        # paths
        farmer_region = utils.get_farmer_region(farmer)
        picture_output = "{picture_t}_output_{ind}".format(picture_t=picture_type, ind=index)
        base_output_folder_farmer = os.path.join(workdir, "Output", picture_output, farmer)
        path_geojsons_folder = path_trainings = os.path.join(workdir, "Meta", farmer, picture_type, "frames")
        path_myproject = os.path.join(agisoft_path, farmer_region, farmer, picture_type)

        # iterate over dates
        dates = os.listdir(path_geojsons_folder)
        for date in dates:
            if utils._check_date_name(date):
                path_current_masks = os.path.join(base_output_folder_farmer, 'mask', date)
                path_grid_corners = os.path.join(workdir, "Meta", farmer, picture_type, "corners")
                path_RGB_date = os.path.join(workdir, farmer_region, farmer, picture_type, date)
                cornersDF = pd.io.parsers.read_csv("{path_grid_c}/{farm}_{dat}_{picture_t}_CameraCornerCoordinates.csv".format(
                    path_grid_c=path_grid_corners, farm=farmer, dat=date, picture_t=picture_type), index_col=0)
                path_geojsons_date = os.path.join(path_geojsons_folder, date)
                path_current_masks = os.path.join(base_output_folder_farmer, 'mask', date)
                path_output_frame_csv = os.path.join(base_output_folder_farmer, "frames", date)
                Path(path_output_frame_csv).mkdir(parents=True, exist_ok=True)

                project = "HF_{farm}_{dat}_{picture_t}.psx".format(farm=farmer, dat=date, picture_t=picture_type)
                doc = Metashape.Document()
                path_project = os.path.join(path_myproject, date, project)
                doc.open(path_project)
                chunk = doc.chunk

                # load all available geojsons
                geojsons = os.listdir(path_geojsons_date)
                for geoj in geojsons:
                    # pic_name= name of the geojson without the .geojson
                    if utils._check_geojson(geoj):
                        path_geojson_current = os.path.join(path_geojsons_date)  ## needs to be adapted!!!!
                        with open("{path_geo}/{pic_n}".format(path_geo=path_geojson_current, pic_n=geoj),
                                  'r') as infile:
                            polygon_mask = geojson.load(infile)
                            # max min pro bild aus cornersDF in x und y richtung finden. Dann koordinaten array testen min < coord1 < max and min< coord2 < max
                            # wenn das True ist, bild nehmen fortfahren, wenn das false ist skipen
                            polygons = polygon_mask["features"]
                            # iterate over the polygons within the geojson file
                            for polygon in polygons:
                                frame_coverage = []
                                # coords = polygon["geometry"]["coordinates"][0][0] #might be needed for 10m?!
                                coords = polygon["geometry"]["coordinates"][0]
                                frame_label = polygon["properties"]["plot_label"]
                                print(frame_label)
                                # find the images of interest
                                # --> only min and max values are considered
                                # --> as the are polygons and not rectangles too many images will be found
                                # --> check afterwards for that
                                images = FrameFunctions.image_finder(cornersDF, coords)
                                for image in images:
                                    img_id = image[0]
                                    # agisoft chunk to translate real word coordinates into image coordinates
                                    # THIS DOES NOT SEEM TO WORK, PROBABLY DUE TO CAMERAS NOT USED IN ALIGNMENT
                                    # camera = chunk.cameras[image[1]]
                                    # Replaced with:
                                    camera = next(camera for camera in chunk.cameras if camera.label == img_id)
                                    sens = [chunk.sensors[0].height, chunk.sensors[0].width]
                                    coords_2dim = copy.deepcopy(coords)

                                    coords_pic = []
                                    for cornerNR in range(0, len(coords_2dim)-1):

                                        out = AgisoftFunctions.coordinates_world2pic_translator(chunk, camera,
                                                                               point_world=coords_2dim[cornerNR])
                                        if out is None:
                                            break
                                            out = [-999, -999]
                                            print("NoneType detected!")
                                        coords_pic.append(out)

                                    if len(coords_pic) > 0:
                                        x = [item[0] for item in coords_pic]
                                        y = [item[1] for item in coords_pic]
                                        c = [x, y]
                                        centr = (sum(c[0]) / len(c[0]), sum(c[1]) / len(c[1]))
                                        centr_x = centr[1]
                                        centr_y = centr[0]
                                    else:
                                        centr_x = "NA"
                                        centr_y = "NA"

                                    if AgisoftFunctions.pic_coordinate_checker(coords_pic, sens):

                                        image_name = "{img}.{pic_form}".format(img=image[0], pic_form=picture_format)
                                        # print(image_name)
                                        path_RGB_image = os.path.join(path_RGB_date, image_name)
                                        RGB_pic = mpimg.imread(str(path_RGB_image))
                                        index_pic = index_function(RGB_pic)
                                        frame_cov = FrameFunctions.polygon_value_calculator(corner_grid_pic=coords_pic,
                                                                                            contour_mask=index_pic)
                                        out = ({'img_id':img_id, 'centr_x':centr_x, 'centr_y':centr_y, 'cover':frame_cov})
                                        frame_coverage.append(out)

                                csv_namer = "{path_out}/{ind}_{label}_{dat}.csv".format(path_out=path_output_frame_csv,
                                                                                        ind=index,
                                                                                        label=frame_label,
                                                                                        dat=date)

                                frame_coverage = pd.DataFrame(frame_coverage)
                                frame_coverage.to_csv(csv_namer)








