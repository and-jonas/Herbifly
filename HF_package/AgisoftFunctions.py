### Author: Flavian Tschurr
### Project: Herbifly
### Script use: provides various functions using agisoft project; delivering coordiantes
# and transaltion into different coordinate systems (world and picture)
### Date: 06.05.2020

########################################################################################################################
# imports
########################################################################################################################

import Metashape
import pandas as pd
import math
import numpy as np
from matplotlib import path
import HF_package.utils as utils


def get_corner_coordinates(chunk):
    """
    calculates for every used image in the agisoft project chunk the coordinates of the corners
    :param chunk: chunk of a agisoft project
    :return: dataframe with the name of the image and for each corner x,y and z coordinates in world coordinate system
     (attention defined as the coordinate system set in agisoft chunk!)
    """
    # name the columns of the output data frame
    cols = ["camera","e1_x","e1_y", "e1_z","e2_x","e2_y", "e2_z","e3_x","e3_y", "e3_z","e4_x","e4_y", "e4_z"]
    # create variables/objects which are used during the calculation
    sensor = chunk.sensors
    # we choose a agisoft model --> prefered the DEM, dense_cloud > point_cloud
    if chunk.model:
        surface = chunk.model
    elif chunk.dense_cloud:
        surface = chunk.dense_cloud
    else:
        surface = chunk.point_cloud
    # we create a new list
    allCorners=list()
    # matrix to transofrm the coordinates from "picture-coordiante system," to world coordinate system (attention world
    # coordinate system is the system defined in the chunk!)
    T = chunk.transform.matrix
    # loop over all pictures
    for camera in chunk.cameras:
        corners = list()
        print(camera.label)
        if camera.center is None:
            print("not used in the chunk")
        else:
            for i in [[0, 0], [sensor[0].width - 1, 0], [sensor[0].width - 1, sensor[0].height - 1], [0, sensor[0].height - 1]]:

                corners.append(chunk.point_cloud.pickPoint(camera.center, camera.unproject(Metashape.Vector(i))))

                if not corners[-1]:
                    print(camera.center)
                    corners[-1] = (surface.pickPoint(camera.reference.location, camera.unproject(Metashape.Vector(i))))
                if not corners[-1]:
                    break
                corners[-1] = chunk.crs.project(T.mulp(corners[-1]))

            allCorners.append([camera.label,corners[0][0],corners[0][1],corners[0][2],corners[1][0],corners[1][1],corners[1][2],corners[2][0],corners[2][1],corners[2][2],corners[3][0],corners[3][1],corners[3][2]])

    cornersDF= pd.DataFrame(allCorners, columns=cols)
    return cornersDF


def get_corner_coordinates_middle_tile(chunk):
    """
    calculates for every used image in the agisoft project chunk the coordinates of the corners
    :param chunk: chunk of a agisoft project
    :return: dataframe with the name of the image and for each corner x,y and z coordinates in world coordinate system
     (attention defined as the coordinate system set in agisoft chunk!)
    """
    # name the columns of the output data frame
    cols = ["camera","e1_x","e1_y", "e1_z","e2_x","e2_y", "e2_z","e3_x","e3_y", "e3_z","e4_x","e4_y", "e4_z"]
    # create variables/objects which are used during the calculation
    sensor = chunk.sensors
    # we choose a agisoft model --> prefered the DEM, dense_cloud > point_cloud
    if chunk.model:
        surface = chunk.model
    elif chunk.dense_cloud:
        surface = chunk.dense_cloud
    else:
        surface = chunk.point_cloud
    # we create a new list
    allCorners=list()
    # matrix to transofrm the coordinates from "picture-coordiante system," to world coordinate system (attention world
    # coordinate system is the system defined in the chunk!)
    T = chunk.transform.matrix
    # loop over all pictures
    for camera in chunk.cameras:
        corners = list()
        print(camera.label)
        if camera.center is None:
            print("not used in the chunk")
        else:
            for i in [[sensor[0].width/3 - 1, sensor[0].height/3 - 1], [2*(sensor[0].width/3) - 1, sensor[0].height/3 - 1], [2*(sensor[0].width)/3 - 1, 2*(sensor[0].height)/3 - 1], [sensor[0].width/3, 2*(sensor[0].height)/3 - 1]]:

                corners.append(chunk.point_cloud.pickPoint(camera.center, camera.unproject(Metashape.Vector(i))))

                if not corners[-1]:
                    print(camera.center)
                    corners[-1] = (surface.pickPoint(camera.reference.location, camera.unproject(Metashape.Vector(i))))
                if not corners[-1]:
                    break
                corners[-1] = chunk.crs.project(T.mulp(corners[-1]))

            allCorners.append([camera.label,corners[0][0],corners[0][1],corners[0][2],corners[1][0],corners[1][1],corners[1][2],corners[2][0],corners[2][1],corners[2][2],corners[3][0],corners[3][1],corners[3][2]])

    cornersDF= pd.DataFrame(allCorners, columns=cols)
    return cornersDF


# best --> calculate this once, then safe and reload (probably into a netcdf --> no confusion will arise
def farmer_grid_creator(cornersDF, gridSize,buffer=20):
    """ creates for a field the "master" grid --> we calculate this once and the us it to reference the other images
    and build a dataframe combining all data per date
    :param cornersDF: data frame with coordinates of the images (we need this to calculate min and max values
    to determine the boarder of the dataframe)
    :param gridSize: the wanted size of the grid in meters (as we use swiss coordinate system!)
    :param buffer: adds a certain distance to every side of the field to ensure all coordinates are within the field. (take a even value)
    :return: two arrays with x and y coordinates --> use to create a df or netCDF
    """
    # we calculate the real world max and min values of the field in the x and y direction (first per corner,
    # then over all corners)
    corners_x_max = [cornersDF.e1_x.max(),cornersDF.e2_x.max(),cornersDF.e3_x.max(),cornersDF.e4_x.max()]
    corners_x_min =[cornersDF.e1_x.min(),cornersDF.e2_x.min(),cornersDF.e3_x.min(),cornersDF.e4_x.min()]

    corners_y_max = [cornersDF.e1_y.max(),cornersDF.e2_y.max(),cornersDF.e3_y.max(),cornersDF.e4_y.max()]
    corners_y_min =[cornersDF.e1_y.min(),cornersDF.e2_y.min(),cornersDF.e3_y.min(),cornersDF.e4_y.min()]

    # max values are allawy round to the next higher int, min to the enxt lower --> to ensure to not lose anything
    field_x_max = math.ceil(max(corners_x_max))
    field_x_min = math.floor(min(corners_x_min))
    field_x_max = utils.round_up_to_even(field_x_max)
    field_x_min = utils.round_down_to_even(field_x_min)

    field_y_max =math.ceil(max(corners_y_max))+buffer
    field_y_min =math.floor(min(corners_y_min))-buffer
    field_y_max = utils.round_up_to_even(field_y_max)+buffer
    field_y_min = utils.round_down_to_even(field_y_min)-buffer

    # calculate the length of the dataframe  --> how many rows /cloumns using the grid size
    x_length = int((field_x_max-field_x_min)/gridSize)
    y_length = int((field_y_max-field_y_min)/gridSize)

    # create for x and y direction a vector with all coordinates
    x_coords =[]
    for x_coord in range(x_length):
        coord = field_x_max - (x_coord*gridSize)
        x_coords.append(coord)

    x_coords = np.array(x_coords)

    y_coords =[]
    for y_coord in range(y_length):
        coord = field_y_max - (y_coord*gridSize)
        y_coords.append(coord)

    y_coords = np.array(y_coords)
    #
    # ## here we create a meshgrid (2 layers, 0 = x values, 1 = y values) of the world coordinates
    # xx, yy = np.meshgrid(x_coords,y_coords, indexing="ij")
    # coordinate_grid = np.array([xx,yy])

    return x_coords, y_coords


def image_grid_creator(cornersDF,gridSize,current_picName):
    """

    :param cornersDF: Dataframe containing all corner Coordiantes iof the used cameras (pictures in the chunk)
    :param gridSize: size of the grid in world coordinates
    :param current_picName: name of the current picture/camera
    :return:
    """

    # we select just the row with the coordinates of the current picture out of the DF
    wantedRow = cornersDF[cornersDF.camera == current_picName]
    pic_x_max = math.ceil(max([int(wantedRow.e1_x) , int(wantedRow.e2_x) , int(wantedRow.e3_x) , int(wantedRow.e4_x)]))
    pic_x_min = math.floor(min([int(wantedRow.e1_x) , int(wantedRow.e2_x) , int(wantedRow.e3_x) , int(wantedRow.e4_x)]))
    pic_x_max = utils.round_up_to_even(pic_x_max)
    pic_x_min = utils.round_down_to_even(pic_x_min)

    pic_y_max = math.ceil(max([int(wantedRow.e1_y) , int(wantedRow.e2_y) , int(wantedRow.e3_y) , int(wantedRow.e4_y)]))
    pic_y_min = math.floor(min([int(wantedRow.e1_y) , int(wantedRow.e2_y) , int(wantedRow.e3_y) , int(wantedRow.e4_y)]))
    pic_y_max = utils.round_up_to_even(pic_y_max)
    pic_y_min = utils.round_down_to_even(pic_y_min)
    # calculate the length of the dataframe  --> how many rows /cloumns using the grid size
    x_length = int((pic_x_max - pic_x_min) / gridSize)+1
    y_length = int((pic_y_max - pic_y_min) / gridSize)+1

    # create for x and y direction a vector with all coordinates
    x_coords = []
    for x_coord in range(x_length):
        pic_x_max_higher = pic_x_max + gridSize
        coord = pic_x_max_higher - (x_coord * gridSize)
        x_coords.append(coord)

    x_coords = np.array(x_coords)

    y_coords = []
    for y_coord in range(y_length):
        pic_y_max_higher = pic_y_max + gridSize
        coord = pic_y_max_higher - (y_coord * gridSize)
        y_coords.append(coord)

    y_coords = np.array(y_coords)

    ## here we create a meshgrid (2 layers, 0 = x values, 1 = y values) of the world coordinates
    # pic_world_x, pic_world_y = np.meshgrid(x_coords,y_coords, sparse=True)
    # coordinate_grid_pic = np.array([pic_x,pic_y])
    return x_coords, y_coords


# function to translate the world coordinates into picture coordinates
def coordinates_world2pic_translator_oldone(chunk, camera,point_world):
    """
    translates coordinates from world into picture coordinates
    :param chunk: current agisoft chunk
    :param camera: current camera
    :param point_world: 2D coordinates of a point in the world (best on the picture ;) )
    :return:
    """
    point_world.append(chunk.elevation.altitude(point_world))
    # ret = camera.project(chunk.transform.matrix.inv().mulp(chunk.crs.unproject(point_world)))

    ret = camera.project(chunk.transform.matrix.mulp(chunk.crs.unproject(point_world)))
    print(ret)
    return ret


def coordinates_world2pic_translator(chunk, camera, point_world):
    """
    translates coordinates from world into picture coordinates
    :param chunk: current agisoft chunk
    :param camera: current camera
    :param point_world: 2D coordinates of a point in the world (best on the picture ;) )
    :return:
    """
    point_world.append(chunk.elevation.altitude(point_world))
    # T = camera.transform
    T = chunk.transform.matrix
    # ret = camera.project(camera.transform.inv().mulp(chunk.crs.unproject(point_world)))

    # ret = camera.project(T.inv().mulp(chunk.crs.unproject(point_world)))

    # unproject a point --> point as Metashape Vector
    unprojected = chunk.crs.unproject(Metashape.Vector(point_world))
    # multiply with the inverse transform matrix (of the chunk!)
    transformed = T.inv().mulp(Metashape.Vector(unprojected))
    # project using camera. project
    ret = camera.project(Metashape.Vector(transformed))
    # print(ret)
    return ret


# check if real world coordinate is within the picture ( using the output of the image_grid_creator)
def pic_coordinate_checker(corner_grid_pic, sens):
    """
    checks if all corners are within the picture
    :param corner_grid_pic: list with corners of the current grid cell in picture coordinate system
    :param sens: sensors (width and height)
    :return: True/false
    """
    counter = 0
    for corner in corner_grid_pic:
        if 1 <= corner[0] <= sens[0]:
            if 1 <= corner[1] <= sens[1]:
                counter = counter + 1
    if counter == 4:
        out = True
    else:
        out = False
    return out


def pic_coordinate_checker_middle_tile(corner_grid_pic, coords_middle):
    """
    checks if all corners are within the picutre
    :param corner_grid_pic: list with corners of the current grid cell in picture coordinate system
    :param sens: sensors (width and height)
    :return: True/false
    """
    counter = 0

    for corner in corner_grid_pic:
        if coords_middle[0][0] <= corner[1] <= coords_middle[0][1]:
            if coords_middle[0][2] <= corner[0] <= coords_middle[0][3]:
                counter = counter + 1
    if counter == 4:
        out = True
    else:
        out = False
    return out


def grid_coverage_calculator(corner_grid_pic, contour_mask):
    """
    :param corner_grid_pic: corners of a grid within a picture
    :param contour_mask: detected contours
    :return: coverage as a ratio
    """

    cm = np.asarray([[1216, 2432, 1824, 3648]])

    # transform coordinates to a path
    grid_path = path.Path(corner_grid_pic, closed=False)

    # create a mask of the image
    xcoords = np.arange(cm[0][0], cm[0][0] + contour_mask.shape[0])
    ycoords = np.arange(cm[0][2], cm[0][2] + contour_mask.shape[1])
    # coords = np.transpose([np.repeat(xcoords, len(ycoords)), np.tile(ycoords, len(xcoords))])
    coords = np.transpose([np.repeat(ycoords, len(xcoords)), np.tile(xcoords, len(ycoords))])

    # Create mask
    grid_mask_image = grid_path.contains_points(coords, radius=-0.0)
    ## check if correct or if the axes needed to be swaped! --> change corresponding line in code
    # grid_mask_image = np.swapaxes(grid_mask_image.reshape(contour_mask.shape[0], contour_mask.shape[1]), 0, 1)
    grid_mask_image = np.swapaxes(grid_mask_image.reshape(contour_mask.shape[1], contour_mask.shape[0]), 0, 1)

    # grid_mask_image = grid_mask_image.reshape(contour_mask.shape[1], contour_mask.shape[0])
    # we take the coordiantes of the polygon --> these are masked now
    polygon_coords = np.argwhere(grid_mask_image == True)
    # we count all pixels which are filled in the contour_mask --> the non 0 values are weeds as we sorted the rest out earlier
    weed_counter = 0
    for pixel in polygon_coords:
        # if contour_mask[pixel[1]][pixel[0]] !=0:
        if contour_mask[pixel[0]][pixel[1]] != 0:
            weed_counter = weed_counter+1
    # we just calcualted the ratio of covered and non covered pixels
    if len(polygon_coords)> 0:
        cover_ratio = weed_counter/len(polygon_coords)
    else:
        # cover_ratio = 99999
        cover_ratio = None
    return cover_ratio


def iterate_pic_grid(x_coords , y_coords , chunk , gridSize , camera , contour_mask):
    """ This function iterates over each cell of the grid and fills it with the according value
    :param x_coords: x coordinate string
    :param y_coords: y coordinate string
    :param chunk: agisoft chunk
    :param gridSize: size of the grid
    :param camera: camera of a agisoft project
    :param contour_mask: created mask with weeds on it
    :return: dataframe in grid format
    """

    sens = [chunk.sensors[0].height , chunk.sensors[0].width]
    # creating an emtpy dataframe with column and row names (x and y coordinates!)
    onePic = []
    # print(f'grid filling for {camera.label} is in process')

    for x in range(0 , len(x_coords)):
        for y in range(0 , len(y_coords)):
            # point_world_2d =[x_coords[x],y_coords[y]]
            corner_grid_world = [[x_coords[x] + 0.5 * gridSize , y_coords[y] + 0.5 * gridSize] ,
                                 [x_coords[x] + 0.5 * gridSize , y_coords[y] - 0.5 * gridSize] ,
                                 [x_coords[x] - 0.5 * gridSize , y_coords[y] - 0.5 * gridSize] ,
                                 [x_coords[x] - 0.5 * gridSize , y_coords[y] + 0.5 * gridSize]]

            corner_grid_pic = []
            for cornerNR in range(0 , len(corner_grid_world)):

                out = coordinates_world2pic_translator(chunk, camera , point_world=corner_grid_world[cornerNR])
                if out is None:
                    out=[-999,-999]
                    print("NoneType detected!")
                # out = [int(out[0]), int(out[1])]
                corner_grid_pic.append(out)

            check = pic_coordinate_checker(corner_grid_pic , sens)
            if check == True:
                # print(check)
                # chose grid point of the image an fill with the correspondig value
                #-> calculating coverage for the grid segment
                cover = grid_coverage_calculator(corner_grid_pic , contour_mask)

            else:
                cover = None
            onePic.append([x_coords[x],y_coords[y],cover])
            # print( f'{x_coords[x]},{y_coords[y]}')
            print("{x_coo},{y_coo}".format(x_coo=x_coords[x],y_coo=y_coords[y]))
    return onePic


def iterate_pic_grid_middle_tile(x_coords, y_coords, chunk , gridSize,
                                 camera, mask):
    """ This function iterates over each cell of the grid and fills it with the according value
    :param x_coords: x coordinate string
    :param y_coords: y coordinate string
    :param chunk: agisoft chunk
    :param gridSize: size of the grid
    :param camera: camera of a agisoft project
    :param contour_mask: created mask with weeds on it
    :return: dataframe in grid format
    """

    #picture coordinates of middle tile
    coords_middle_tile = np.asarray([[1216, 2432, 1824, 3648]])

    # creating an emtpy dataframe with column and row names (x and y coordinates!)
    onePic = []
    print('>>Calculating coverage, filling grid...')
    all_c_world = []
    corners = []
    checks = []
    for x in range(0 , len(x_coords)):
        for y in range(0 , len(y_coords)):
            # print(f'{x},{y}')
            # point_world_2d =[x_coords[x],y_coords[y]]
            corner_grid_world = [[x_coords[x] + 0.5 * gridSize, y_coords[y] + 0.5 * gridSize],
                                 [x_coords[x] + 0.5 * gridSize, y_coords[y] - 0.5 * gridSize],
                                 [x_coords[x] - 0.5 * gridSize, y_coords[y] - 0.5 * gridSize],
                                 [x_coords[x] - 0.5 * gridSize, y_coords[y] + 0.5 * gridSize]]
            # all_c_world.append(corner_grid_world)
            # b = [item for sublist in all_c_world for item in sublist]
            # l1 = min([item[0] for item in b])
            # l2 = min([item[1] for item in b])
            # l3 = max([item[0] for item in b])
            # l4 = max([item[1] for item in b])
            # # with open("O:/Hiwi/2020_Herbifly/bin/Bb_test.txt", "wb") as fp:  # Pickling
            #     pickle.dump(all_c_world, fp)
            # corners.append(corner_grid_world)

            corner_grid_pic = []
            for cornerNR in range(0, len(corner_grid_world)):
                out = coordinates_world2pic_translator(chunk, camera,
                                                       point_world=corner_grid_world[cornerNR])
                if out is None:
                    out=[-999,-999]
                    print("NoneType detected!")
                # out = [int(out[0]), int(out[1])]
                corner_grid_pic.append(out)
            corners.append(corner_grid_pic)
            # with open("O:/Hiwi/2020_Herbifly/bin/Bb_test_obia.txt", "wb") as fp:  # Pickling
            #     pickle.dump(corners, fp)

            check = pic_coordinate_checker_middle_tile(corner_grid_pic , coords_middle_tile)
            if check == True:
                # chose grid point of the image an fill with the correspondig value
                #-> calculating coverage for the grid segment
                cover = grid_coverage_calculator(corner_grid_pic, mask)

            else:
                cover = None
            onePic.append([x_coords[x], y_coords[y], cover])

    return onePic
