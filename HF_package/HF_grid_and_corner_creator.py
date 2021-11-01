### Author: Flavian Tschurr
### Project: Herbifly
### Date: 05.05.2020

########################################################################################################################
# imports
########################################################################################################################

import Metashape
import pandas as pd
from pathlib import Path
import os

native_os_path_join = os.path.join
def modified_join(*args, **kwargs):
    return native_os_path_join(*args, **kwargs).replace('\\', '/')
os.path.join = modified_join

import utils as utils
import AgisoftFunctions as AgisoftFunctions

########################################################################################################################
# variables
########################################################################################################################

# workdir = Path('../').resolve()
workdir = "O:/Evaluation/Hiwi/2020_Herbifly/Processed_Campaigns"
workdir_out = "O:/Evaluation/Hiwi/2020_Herbifly/Images_Farmers"
picture_type = "30m"
gridSize = 0.5
pic_format = ".JPG"
farmers = ["Baumberger2", "Baumberger1", "Stettler", "Egli", "Scheidegger", "Keller", "Bolli", "Bonny","Miauton"]
# farmers = ["Baumberger2", "Baumberger1", "Stettler", "Egli", "Keller", "Bolli", "Bonny"] #30m
# farmers = ["Baumberger2", "Baumberger1", "Egli", "Keller", "Bolli"] #30m frames
farmers = ["Baumberger1"]

########################################################################################################################
# start of the calculation and iteration over the farmer etc.
########################################################################################################################

for farmer in farmers:

    # create paths
    farmer_region = utils.get_farmer_region(farmer)
    path_myfarm = os.path.join(workdir,farmer_region,farmer, picture_type)
    path_grid_output =os.path.join(workdir_out, "Meta",farmer, picture_type)
    path_grid_output_grids=os.path.join(path_grid_output,"grids")
    path_grid_output_corners=os.path.join(path_grid_output,"corners")
    Path(path_grid_output_grids).mkdir(parents=True, exist_ok=True)
    Path(path_grid_output_corners).mkdir(parents=True, exist_ok=True)
    dates = os.listdir(path_myfarm)

    counter = 0
    for date in dates:

        # check if the date is really a date
        if utils._check_date_name(date):
            counter = counter + 1
            # project = f'HF_{farmer}_{date}_{picture_type}.psx'
            project= "HF_{farm}_{dat}_{picture_t}.psx".format(farm=farmer, dat=date, picture_t=picture_type)
            doc = Metashape.Document()
            path_project = os.path.join(path_myfarm, date, project)
            doc.open(path_project)
            chunk = doc.chunk
            if picture_type == '30m':
                cornersDF = AgisoftFunctions.get_corner_coordinates(chunk)  # for 30m
            elif picture_type == '10m':
                cornersDF = AgisoftFunctions.get_corner_coordinates_middle_tile(chunk)  # for 10m
            if counter == 1:
                x_coords, y_coords = AgisoftFunctions.farmer_grid_creator(cornersDF, gridSize)
                oneFieldGrid = pd.DataFrame(data=None, columns=x_coords, index=y_coords)
                gridIndication = round(gridSize * 100)
                oneFieldGrid.to_csv("{path_grid_output_g}/{farm}_{picture_t}_FieldGrid_{gridIndi}.csv".format(
                    path_grid_output_g=path_grid_output_grids, farm=farmer, picture_t=picture_type,
                    gridIndi=gridIndication))



