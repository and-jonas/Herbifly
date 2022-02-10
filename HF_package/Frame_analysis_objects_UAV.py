
# ======================================================================================================================
# Author: Flavian Tschurr
# Project: Herbifly
# Script use: wrapper script to analyse frames (as geojson) from an orthomosaic on single images
# This wrapper uses the output masks of the HF_segmentation_main!! It is not a standalone script, no segmentation and
# classification is applied to reduce comnputing time and bias from different segmentation results
# Date: 17.06.2020
# Last edited: Jonas Anderegg, 2021-11-03
# ======================================================================================================================

# imports
import pandas as pd
import numpy as np
import os

## the following lines are to change de default separator within paths (if some strange behaviour in built path occurs, check here!)
native_os_path_join = os.path.join
def modified_join(*args, **kwargs):
    return native_os_path_join(*args, **kwargs).replace('\\', '/')
os.path.join = modified_join

import geojson
import Metashape
from PIL import Image
from pathlib import Path
import copy

import HF_package.utils as utils
import HF_package.FrameFunctions as FrameFunctions
import HF_package.AgisoftFunctions as AgisoftFunctions

# ======================================================================================================================

farmers = ["Baumberger1", "Baumberger2", "Stettler", "Egli", "Scheidegger", "Keller", "Bolli", "Bonny", "Miauton"]
farmers = ["Bonny"]
agisoft_path= "O:/Hiwi/2020_Herbifly/Processed_Campaigns"
workdir = "O:/Hiwi/2020_Herbifly/Images_Farmers"
picture_type = "10m"
features = "all"
picture_roi = "fullsize"
picture_format = "JPG"

# ======================================================================================================================
# start of the calculation
# ======================================================================================================================

for farmer in farmers:

    farmer_region = utils.get_farmer_region(farmer)
    picture_output = "{picture_t}_output".format(picture_t=picture_type)
    base_output_folder_farmer = os.path.join(workdir, "Output", picture_output, features, farmer)
    path_geojsons_folder = path_trainings = os.path.join(workdir, "Meta", farmer, picture_type, "frames")
    path_myproject = os.path.join(agisoft_path, farmer_region, farmer, picture_type)

    dates = os.listdir(path_geojsons_folder)
    # exclude late flights
    dates = [x for x in dates if "202007" not in x]
    dates = [x for x in dates if "202008" not in x]

    for date in dates:
        try:
            if utils._check_date_name(date):
                path_grid_corners = os.path.join(workdir, "Meta", farmer, picture_type, "corners")
                path_RGB_date = os.path.join(workdir, farmer_region, farmer, picture_type, date)
                cornersDF = pd.io.parsers.read_csv("{path_grid_c}/{farm}_{dat}_{picture_t}_{pic_roi}_CameraCornerCoordinates.csv".format(
                    path_grid_c=path_grid_corners, farm=farmer, dat=date, picture_t=picture_type, pic_roi=picture_roi),
                    index_col=0)
                path_geojsons_date = os.path.join(path_geojsons_folder, date)
                path_current_masks = os.path.join(base_output_folder_farmer, 'prediction', date)
                path_output_frame_csv = os.path.join(base_output_folder_farmer, "frames", date)
                Path(path_output_frame_csv).mkdir(parents=True, exist_ok=True)

                project = "HF_{farm}_{dat}_{picture_t}.psx".format(farm=farmer, dat=date, picture_t=picture_type)
                doc = Metashape.Document()
                path_project = os.path.join(path_myproject, date, project)
                doc.open(path_project)
                chunk = doc.chunk

                # iterate over geojsons
                geojsons = os.listdir(path_geojsons_date)
                for geoj in geojsons:
                    if utils._check_geojson(geoj):
                        path_geojson_current = os.path.join(path_geojsons_date)
                        with open("{path_geo}/{pic_n}".format(path_geo=path_geojson_current, pic_n=geoj),
                                  'r') as infile:
                            polygon_mask = geojson.load(infile)
                            polygons = polygon_mask["features"]
                            # iterate over the polygons within the geojson file
                            for polygon in polygons:
                                frame_coverage = []
                                # format of geojsons seems to be inconsistent
                                if len(polygon["geometry"]["coordinates"][0]) == 5:
                                    coords = polygon["geometry"]["coordinates"][0]
                                else:
                                    coords = polygon["geometry"]["coordinates"][0][0]
                                frame_label = polygon["properties"]["plot_label"]
                                print(frame_label)
                                # find the images of interest
                                # --> attention to make this step "easy" just min and max values are considered
                                # --> as the are polygons and not rectangles too many images will be found
                                # --> we check afterwards for that
                                images = FrameFunctions.image_finder(cornersDF, coords)
                                for image in images:
                                    img_id = image[0]
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
                                        path_countour_mask = "{path_m}/{img}_predicted_mask.tif".format(
                                                path_m=path_current_masks, img=image[0])
                                        try:
                                            contour_mask = Image.open(path_countour_mask)
                                            contour_mask = np.array(contour_mask)
                                            frame_cov = AgisoftFunctions.grid_coverage_calculator(coords_pic, contour_mask)

                                            out = ({'img_id': img_id, 'centr_x': centr_x,
                                                    'centr_y': centr_y, 'cover': frame_cov})
                                            frame_coverage.append(out)
                                        except FileNotFoundError:
                                            print("x")

                                csv_namer = "{path_out}/{label}_{dat}.csv".format(path_out=path_output_frame_csv,
                                                                                  label=frame_label, dat=date)

                                frame_coverage = pd.DataFrame(frame_coverage)
                                frame_coverage.to_csv(csv_namer)
        except FileNotFoundError:
            print(f'A file not found - skipping date: {date}')
            continue

