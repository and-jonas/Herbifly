# Author: Lukas Roth, lukas.roth@usys.ethz.ch

import rawpy
import cv2
import numpy as np
# import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import path
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
import geojson
from descartes import PolygonPatch



def preview_raw_image(image_raw, pixel_shift=0):
    '''Gets a rawpy.RawPy object and generates a 8bit preview tiff

    :param image_raw: rawpy.RawPy object
    :param pixel_shift: Offset pixel used to cutout border pixel
    :return: 8bit RGB array
    '''

    # Use linear interpolation as demosaic algorithm with the library rawpy
    image_8bit = image_raw.postprocess(
        output_bps=8,
        output_color=rawpy.ColorSpace.sRGB,
        demosaic_algorithm=0)

    return (image_8bit[
                pixel_shift:image_8bit.shape[0] - pixel_shift,
                pixel_shift:image_8bit.shape[1] - pixel_shift
            ])

def demosaic_raw_image(image_raw, pixel_shift=0):
    '''Gets a rawpy.RawPy object and generates 16 bit RGB, HSV, Lab and ExG/ExR representants

    :param image_raw: rawpy.RawPy object
    :param pixel_shift: Offset pixel used to cutout border pixel
    :return: tuple of RGB, HSV, and Lab (all 0..1 float32) and ExG and ExR (n..m float32)
    :return: descriptors as numpy array
    :return: descriptor names
    '''

    # Demosaic image using rawpy
    a_XYZ_16bit = image_raw.postprocess(
        output_bps=16,
        output_color=rawpy.ColorSpace.XYZ,
        demosaic_algorithm=0,
        use_camera_wb=True,
        no_auto_bright=True)

    # Cut image
    a_XYZ_16bit = a_XYZ_16bit[
                pixel_shift:a_XYZ_16bit.shape[0] - pixel_shift,
                pixel_shift:a_XYZ_16bit.shape[1] - pixel_shift,
                :
            ]

    # Convert XYZ to RGB space using opencv (cv2)
    a_RGB_16bit = cv2.cvtColor(a_XYZ_16bit, cv2.COLOR_XYZ2RGB)

    # Scale to 0...1
    a_RGB_16bitf = np.array(a_RGB_16bit / 2 ** 16, dtype=np.float32)

    # Convert to HSV space using opencv (cv2)
    a_HSV_16bitf = cv2.cvtColor(a_RGB_16bitf, cv2.COLOR_RGB2HSV)

    # Convert to LAB space using opencv (cv2)
    a_Lab_16bitf = cv2.cvtColor(a_RGB_16bitf, cv2.COLOR_RGB2Lab)

    # Calcualte vegetation indices: ExR and ExG
    R, G, B = cv2.split(a_RGB_16bit)
    normalizer = np.array(R + G + B, dtype=np.float32)
    # Avoid division by zero
    normalizer[normalizer == 0] = 1
    r, g, b = (R, G, B) / normalizer
    # ExR + ExG
    a_ExR = np.array(1.4 * r - b, dtype=np.float32)
    a_ExG = np.array(2.0 * g - r - b, dtype=np.float32)

    # Concat all
    descriptors = np.concatenate(
        [a_XYZ_16bit, a_RGB_16bitf, a_HSV_16bitf, a_Lab_16bitf, np.stack([a_ExG, a_ExR], axis=2)], axis=2)
    # Names
    descriptor_names = ['X', 'Y', 'Z', 'sR', 'sG', 'sB', 'H', 'S', 'V', 'L', 'a', 'b', 'ExR', 'ExG']

    # Return as tuple
    return ( (a_XYZ_16bit, a_RGB_16bitf, a_HSV_16bitf, a_Lab_16bitf, a_ExG, a_ExR), descriptors, descriptor_names )

def demosaic_8bit_image(a_8bit_img, pixel_shift=0):
    '''Gets a rawpy.RawPy object and generates 16 bit RGB, HSV, Lab and ExG/ExR representants


    :param pixel_shift: Offset pixel used to cutout border pixel
    :return: tuple of RGB, HSV, and Lab (all 0..1 float32) and ExG and ExR (n..m float32)
    :return: descriptors as numpy array
    :return: descriptor names
    '''

    a_XYZ_8bit = a_8bit_img
    # Cut image
    a_XYZ_8bit = a_XYZ_8bit[
                pixel_shift:a_XYZ_8bit.shape[0] - pixel_shift,
                pixel_shift:a_XYZ_8bit.shape[1] - pixel_shift,
                :
            ]

    # Convert XYZ to RGB space using opencv (cv2)
    a_RGB_8bit = cv2.cvtColor(a_XYZ_8bit, cv2.COLOR_XYZ2RGB)

    # # =====
    # a_RGB_8bit = img
    # # =====

    # Scale to 0...1 ----> das wird failen! float anpassne?
    # a_RGB_8bitf = np.array(a_RGB_8bit / 2 ** 16, dtype=np.float32)
    a_RGB_8bitf = np.array(a_RGB_8bit / 2 ** 8, dtype=np.float32)


    ll = a_RGB_8bitf[:, :, 0]

    # Convert to HSV space using opencv (cv2)
    a_HSV_8bitf = cv2.cvtColor(a_RGB_8bitf, cv2.COLOR_RGB2HSV)

    # Convert to LAB space using opencv (cv2)
    a_Lab_8bitf = cv2.cvtColor(a_RGB_8bitf, cv2.COLOR_RGB2Lab)

    # Calcualte vegetation indices: ExR and ExG
    R, G, B = cv2.split(a_RGB_8bit)
    normalizer = np.array(R + G + B, dtype=np.float32)
    # Avoid division by zero
    normalizer[normalizer == 0] = 1
    r, g, b = (R, G, B) / normalizer
    # ExR + ExG
    a_ExR = np.array(1.4 * r - b, dtype=np.float32)
    a_ExG = np.array(2.0 * g - r - b, dtype=np.float32)

    # fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    # # Show RGB and segmentation mask
    # axs[0].imshow(img)
    # axs[0].set_title('original patch')
    # axs[1].imshow(a_ExG, vmin=-0.5, vmax=1)
    # axs[1].set_title('entropy')
    # plt.show(block=True)

    # Concat all
    descriptors = np.concatenate(
        [a_XYZ_8bit, a_RGB_8bitf, a_HSV_8bitf, a_Lab_8bitf, np.stack([a_ExG, a_ExR], axis=2)], axis=2)
    # Names
    descriptor_names = ['X', 'Y', 'Z', 'sR', 'sG', 'sB', 'H', 'S', 'V', 'L', 'a', 'b', 'ExR', 'ExG']

    # Return as tuple
    return ( (a_XYZ_8bit, a_RGB_8bitf, a_HSV_8bitf, a_Lab_8bitf, a_ExG, a_ExR), descriptors, descriptor_names )

def get_color_spaces(img):

    a_RGB_8bit = img

    # Scale to 0...1 ----> das wird failen! float anpassne?
    # a_RGB_8bitf = np.array(a_RGB_8bit / 2 ** 16, dtype=np.float32)
    a_RGB_8bitf = np.array(a_RGB_8bit / 2 ** 8, dtype=np.float32)

    # Convert to HSV space using opencv (cv2)
    a_HSV_8bitf = cv2.cvtColor(a_RGB_8bitf, cv2.COLOR_RGB2HSV)

    # Convert to LAB space using opencv (cv2)
    a_Lab_8bitf = cv2.cvtColor(a_RGB_8bitf, cv2.COLOR_RGB2Lab)

    # Calcualte vegetation indices: ExR and ExG
    R, G, B = cv2.split(a_RGB_8bitf)
    normalizer = np.array(R + G + B, dtype=np.float32)
    # Avoid division by zero
    normalizer[normalizer == 0] = 10

    r, g, b = (R, G, B) / normalizer

    # ExR + ExG
    a_ExR = np.array(1.4 * r - b, dtype=np.float32)
    a_ExG = np.array(2.0 * g - r - b, dtype=np.float32)

    # fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    # # Show RGB and segmentation mask
    # axs[0].imshow(img)
    # axs[0].set_title('original patch')
    # axs[1].imshow(a_ExG, vmin=-0.5, vmax=1)
    # axs[1].set_title('entropy')
    # plt.show(block=True)

    # Concat all
    descriptors = np.concatenate(
        [a_RGB_8bitf, a_HSV_8bitf, a_Lab_8bitf, np.stack([a_ExG, a_ExR], axis=2)], axis=2)
    # Names
    descriptor_names = ['sR', 'sG', 'sB', 'H', 'S', 'V', 'L', 'a', 'b', 'ExG', 'ExR']

    # Return as tuple
    return ( (a_RGB_8bitf, a_HSV_8bitf, a_Lab_8bitf, a_ExG, a_ExR), descriptors, descriptor_names )

def extract_training_from_coordinates(descriptors, training_coordinates):
    '''Extract per coordinate pixel values from color spaces

    :param descriptors: RGB, HSV, Lab, ExG and ExR as 3 dimensional numpy array
    :param coordinates: Coordinates of training in the form ([[x1, y1],...,[xn, yn]], [[x1, y1],...,[xn, yn]])
    :return: (training, response)
    '''
    training = []
    response = []

    # Sample image per coordinate
    for sample in training_coordinates:
        # Round coordinates
        x_image, y_image = int(round(sample['x'])), int(round(sample['y']))

        # Sample
        training_ = descriptors[y_image, x_image].tolist()
        response_ = 0 if sample['set'] == 'soil' else 1

        # Append to training set
        training.append(training_)
        response.append(response_)

    # Convert to numpy array
    a_training = np.array(training)
    a_response = np.array(response)

    return(a_training, a_response)

def plot_color_spaces(a_RGB_8bit, a_RGB_16bitf, a_HSV_16bitf, a_Lab_16bitf, a_ExG, a_ExR, subsection):
    '''Plots all color spaces as nice 3d plots

    :param a_RGB_8bit: RGB as 0...255 uint8
    :param a_RGB_16bitf: RGB as 0..1 float32
    :param a_HSV_16bitf: HSV as 0..1 float32
    :param a_Lab_16bitf: Lab as 0..1 float32
    :param a_ExG: ExG as n..m float32
    :param a_ExR: ExR as n..m float32
    :param subsection: tuple with length=4 defining a region x1 - x2, y1 - y2 to use for plot
    '''

    fig = plt.figure(figsize=(20, 5))

    ax = fig.add_subplot(141, projection='3d')
    ax.scatter(a_RGB_16bitf[subsection[0]:subsection[1], subsection[2]:subsection[3], 0],
               a_RGB_16bitf[subsection[0]:subsection[1], subsection[2]:subsection[3], 1],
               a_RGB_16bitf[subsection[0]:subsection[1], subsection[2]:subsection[3], 2],
               c=a_RGB_8bit[subsection[0]:subsection[1], subsection[2]:subsection[3]].reshape(-1, 3) / 255, s=60)
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    ax.set_title('RGB')
    ax.view_init(30, 185)

    ax = fig.add_subplot(142, projection='3d')
    ax.scatter(a_HSV_16bitf[subsection[0]:subsection[1], subsection[2]:subsection[3], 1],
               a_HSV_16bitf[subsection[0]:subsection[1], subsection[2]:subsection[3], 0],
               a_HSV_16bitf[subsection[0]:subsection[1], subsection[2]:subsection[3], 2],
               c=a_RGB_8bit[subsection[0]:subsection[1], subsection[2]:subsection[3]].reshape(-1, 3) / 255, s=60)
    ax.set_xlabel('S')
    ax.set_ylabel('H')
    ax.set_zlabel('V')
    ax.set_title('HSV')
    ax.view_init(30, 185)

    ax = fig.add_subplot(143, projection='3d')
    ax.scatter(a_Lab_16bitf[subsection[0]:subsection[1], subsection[2]:subsection[3], 0],
               a_Lab_16bitf[subsection[0]:subsection[1], subsection[2]:subsection[3], 1],
               a_Lab_16bitf[subsection[0]:subsection[1], subsection[2]:subsection[3], 2],
               c=a_RGB_8bit[subsection[0]:subsection[1], subsection[2]:subsection[3]].reshape(-1, 3) / 255, s=60)
    ax.set_xlabel('L')
    ax.set_ylabel('a')
    ax.set_zlabel('b')
    ax.set_title('Lab')
    ax.view_init(30, 185)

    ax = fig.add_subplot(144)
    #ax.scatter(a_ExG[subsection[0]:subsection[1], subsection[2]:subsection[3]].reshape(-1), a_ExR[subsection[0]:subsection[1], subsection[2]:subsection[3]].reshape(-1))
    h = ax.hist2d(a_ExG[subsection[0]:subsection[1], subsection[2]:subsection[3]].reshape(-1),
               a_ExR[subsection[0]:subsection[1], subsection[2]:subsection[3]].reshape(-1),
              bins=100, normed=True, range=[[-0.12, 0.08], [0.1, 0.3]])
    ax.set_xlabel('ExG')
    ax.set_ylabel('ExR')
    ax.set_title('ExG and ExR')
    plt.colorbar(h[3], ax=ax)

    plt.show()

def write_geojson_polygon_mask(corners, plot_label, image_name, path_folder):
    '''Writes a geojson mask file containing one single polygon

    :param corners: Corners of polygon as list, first and last entry should be equal to recieve a closed polygon
    :param plot_label: Label of the plot according to CroPyDB standards
    :param image_name: Name of the image the mask belongs to, no suffix (e.g. no '.CR2')
    :param path_folder: Path to the folder to store the file
    '''

    if corners:
        polygon = geojson.Polygon([corners])
        feature = geojson.Feature(geometry=polygon,
                                  properties={'image': image_name, 'plot_label': plot_label, 'type': 'soil'})
        feature_collection = geojson.FeatureCollection(features=[feature])
        with open(path_folder / (image_name + '.geojson'), 'w') as outfile:
            geojson.dump(feature_collection, outfile, indent='\t')
    else:
        print('Empty corners list, nothing to write')

def write_geojson_polygon_mask_handheld(corners, image_name, path_folder):
    '''Writes a geojson mask file containing one single polygon

    :param corners: Corners of polygon as list, first and last entry should be equal to recieve a closed polygon
    :param plot_label: Label of the plot according to CroPyDB standards
    :param image_name: Name of the image the mask belongs to, no suffix (e.g. no '.CR2')
    :param path_folder: Path to the folder to store the file
    '''

    if corners:
        polygon = geojson.Polygon([corners])
        feature = geojson.Feature(geometry=polygon,
                                  properties={'image': image_name, 'type': 'soil'})
        feature_collection = geojson.FeatureCollection(features=[feature])
        # with open(f'{path_folder}/{image_name}.geojson', "w") as outfile:
        with open("{path_f}/{image_n}.geojson".format(path_f=path_folder,image_n=image_name), 'w') as outfile:
            geojson.dump(feature_collection, outfile, indent='\t')
    else:
        print('Empty corners list, nothing to write')

def zonal_stat(polygon_mask, a_segmented, plot_figure=False):
    '''Calculates statistical values on a zone
    
    :param polygon_mask: zone as polygon
    :param a_segmented: raster'''

    # Container to collect data
    data = []

    # Get features in mask
    samples_polygons = polygon_mask['features']

    # Iterate over polygons, sample each polygon
    for polygon in samples_polygons:
        # Get geometry
        coords = polygon['geometry']['coordinates'][0]
        plot_label = polygon['properties']['plot_label']

        print('Sample no', plot_label)

        # Transform coordinates to path
        sample_path = path.Path(coords)

        # Get extent
        xmin, ymin, xmax, ymax = sample_path.get_extents().extents

        # Create mask of plot
        # Create coordinate matrix to check if image pixel in plot polygon
        x_coords = np.arange(0, a_segmented.shape[1])
        y_coords = np.arange(0, a_segmented.shape[0])
        coords = np.transpose([np.repeat(x_coords, len(y_coords)), np.tile(y_coords, len(x_coords))])

        # Create mask
        sample_mask_image = sample_path.contains_points(coords, radius=1)
        sample_mask_image = np.swapaxes(sample_mask_image.reshape(a_segmented.shape[1], a_segmented.shape[0]), 0, 1)

        # Mask zone of raster
        zoneraster = np.ma.masked_array(a_segmented, np.logical_not(sample_mask_image))

        # Calculate statistics of zones
        count = zoneraster.count()
        mean = np.mean(zoneraster)

        std = np.std(zoneraster)
        var = np.var(zoneraster)
        values_percentiles = np.percentile(zoneraster, np.arange(0, 101))

        # add to data container
        value_stat = {'plot_label': plot_label, 'count': count,
                      'mean': mean, 'std': std, 'var': var}

        if plot_figure:
            plt.figure()
            plt.interactive(False)
            plt.imshow(zoneraster)
            plt.title(plot_label + ', Canopy coverage: ' + str(round(mean*100)))
            plt.show()

        data.append({**value_stat})

    data_pd = pd.DataFrame(data, columns=data[0].keys())
    return(data_pd)

def capture_plot_shape_GUI(a_RGB_8bit):
    '''GUI where the user can draw the plot shape on top of a RGB image

    :param a_RGB_8bit: RGB image to use as background
    :return: Corners of the plot shape in form [[x1, y1],...,[xn, yn]]
    '''

    plt.interactive(True)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    tb = plt.get_current_fig_manager().toolbar

    # List of plot corner coordinates
    corners = []

    # Function to capture plot corner coordinates
    def get_corner(event):
        if tb.mode == '':
            # On left mouse click (button==1)
            if event.button == 1:
                print('Captured coordinates:', event.xdata, event.ydata)
                # Add corner coordinates to list
                corners.append([event.xdata, event.ydata])
                # Add point to plot
                fig.canvas.draw()
                ax.scatter('x', 'y', data=pd.DataFrame({'x': [event.xdata], 'y': [event.ydata]}), marker='+', color='white',
                           s=50)

    # Plot image
    ax.imshow(a_RGB_8bit)
    ax.set_title('Use left mouse button to set corners. Close window if finished')
    # Handler mouse click events
    fig.canvas.mpl_connect('button_press_event', get_corner)

    # Start GUI
    plt.interactive(True)
    plt.show(block=True)
    plt.interactive(False)

    # Add first point as last point to close polygon
    if len(corners)>0:
        corners.append(corners[0])
        return(corners)
    else:
        return(None)

def capture_training_positions_GUI(a_RGB_8bit, a_segmented=None, training_coordinates=[], polygon_mask=None):
    '''GUI where the user can select image coordinates as training

    :param a_RGB_8bit: RGB image to use as background
    :param a_segmented: Segmented image with 1: plant, 0: soil
    :param training_coordinates: Coordinates of existing training in the form ([[x1, y1],...,[xn, yn]], [[x1, y1],...,[xn, yn]])
    :param subsection: tuple with length=4 defining a region x1 - x2, y1 - y2 to use for plot
    :return: Coordinates of plant and soil pixels in the form ([[x1, y1],...,[xn, yn]], [[x1, y1],...,[xn, yn]])
    '''

    # List for plot elements
    if a_segmented is not None:
        fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
    else:
        fig, axs = plt.subplots(1, 1)
        axs = [axs]

    # toolbar
    tb = plt.get_current_fig_manager().toolbar

    # Show RGB
    axs[0].imshow(a_RGB_8bit)
    axs[0].set_title('Left mouse button to select plants, right for soil, middle to remove last point. Close window if finished')

    # Show plot shape
    if polygon_mask is not None:
        polygon_patchs = []
        for polygon in polygon_mask['features']:
            polygon_patch = PolygonPatch(polygon['geometry'])
            polygon_patchs.append(polygon_patch)
        patch_collection = mpl.collections.PatchCollection(polygon_patchs, facecolor='none', edgecolor='white',
                                                           linewidth=1)
        axs[0].add_collection(patch_collection)

    if a_segmented is not None:
        # Overlay RGB with segmented image
        mask = np.array(np.repeat(a_segmented[:, :, np.newaxis], 3, axis=2), dtype=np.bool)
        rgb_plants = np.ma.masked_array(a_RGB_8bit, mask)
        rgb_soil = np.ma.masked_array(a_RGB_8bit, ~ mask)
        # Show segmented images
        axs[1].imshow(rgb_plants.filled())
        axs[2].imshow(rgb_soil.filled())

    # Drawing functions for redraw
    def draw_training_points():
        # Redraw all
        fig.canvas.draw()

        # Marked positions
        for plot_id in range(len(axs)):
            if len(training_coordinates) > 0:
                # Get coordinates and split in plant and soil samples
                df_training_coords = pd.DataFrame(training_coordinates)
                splitter = df_training_coords['set'] == 'plant'
                df_training_plants = df_training_coords[splitter]
                df_training_soil = df_training_coords[~splitter]
                # Plot crosses for positions
                axs[plot_id].scatter('x', 'y', data=df_training_plants, marker='+', color='white')
                axs[plot_id].scatter('x', 'y', data=df_training_soil, marker='+', color='red')

    draw_training_points()

    # Event function on click: add or delete training points
    def onclick(event):

        if tb.mode == '':

            # Coordinates of click
            x = event.xdata
            y = event.ydata

            # Button 1: Training point for plants added
            if event.button == 1:
                training_coordinates.append({'x':x, 'y':y, 'set':'plant'})
            # Button 3: Training point for soil added
            elif event.button == 3:
                training_coordinates.append({'x':x, 'y':y, 'set':'soil'})
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
