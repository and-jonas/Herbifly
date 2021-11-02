# initiate the class and use the writen functions

from pathlib import Path

#
# like this you can install wheel files if pip is not working
import pip

#
# from pip._internal import main as pipmain
# def install_whl(path):
#     pipmain(['install', path])
#
# # install_whl("T:/opencv_python-3.4.2.16-cp27-cp27m-win_amd64.whl")
#
# install_whl("C:/Users/anjonas/Downloads/Metashape-1.6.3-cp35.cp36.cp37-none-win_amd64.whl")
import Metashape
# Metashape.License().activate("EAU39-HMJHH-62AAD-23U93-DS8MA")
# print(Metashape.app.activated)

from HF_package.HF_segmentation_main_middle_tile import SegmentationCalculator


def run(workdir, pic_format=None):
    if not pic_format:
        pic_format = '.JPG'
    picture_type = "Handheld"
    picture_roi = ""
    features = "reconstruct"

    farmers = ["Bolli"]
    gridSize = 0.5
    agisoft_path = "O:/Evaluation/Hiwi/2020_Herbifly/Processed_Campaigns"
    segmentation_calculator = SegmentationCalculator(workdir, picture_type, picture_roi,
                                                     pic_format, features, farmers, gridSize, agisoft_path)
    segmentation_calculator.iterate_farmers()
    # segmentation_calculator.iterate_farmers_postharvest(dates=['20200730'])


if __name__ == '__main__':
    # workdir = Path('../').resolve()
    workdir = "O:/Evaluation/Hiwi/2020_Herbifly/Images_Farmers"
    run(workdir)


