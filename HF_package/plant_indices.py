
# ======================================================================================================================
# Author: Jonas Anderegg, Flavian Tschurr
# Project: Herbifly
# Define functions for calculation of plant indices
# Date: 05.06.2020
# Last edited: Jonas Anderegg, 2021-11-01
# ======================================================================================================================

import numpy as np
import cv2


def index_GLI(pictureCurrent):

    R, G, B = cv2.split(pictureCurrent)
    normalizer = np.array(R + G + B, dtype=np.float32)
    # Avoid division by zero
    normalizer[normalizer == 0] = 1
    r, g, b = (R, G, B) / normalizer

    GLI = np.array((2 * g - r - b) / (2 * g + r + b), dtype=np.float32)
    return GLI


def index_TGI(pictureCurrent):

    R, G, B = cv2.split(pictureCurrent)

    normalizer = np.array(R + G + B, dtype=np.float32)
    # Avoid division by zero
    normalizer[normalizer == 0] = 1
    r, g, b = (R, G, B) / normalizer

    lambda_R = 670
    lambda_G = 550
    lambda_B = 480

    # 201810125DMarkII, EOS400D
    # lambdaRed = 600;
    # lambdaGreen = 540;
    # lambdaBlue = 460;
    rTGI = -0.5 * ((lambda_R - lambda_B) * (r - g) - (lambda_R - lambda_G) * (r - b))
    # = -70 * (R-G) - 60(R - B) = 60 * B + 70 * G - 130 * R
    return rTGI


def index_VEG(pictureCurrent):

    R, G, B = cv2.split(pictureCurrent)

    normalizer = np.array(R + G + B, dtype=np.float32)
    # Avoid division by zero
    normalizer[normalizer == 0] = 1
    r, g, b = (R, G, B) / normalizer

    # Avoid division by zero
    r[r == 0] = 0.00001
    b[b == 0] = 0.00001

    rVEG = g/((r**0.667)*(b**0.333))
    # = -70 * (R-G) - 60(R - B) = 60 * B + 70 * G - 130 * R
    return rVEG


def index_ExG(pictureCurrent):

    R, G, B = cv2.split(pictureCurrent)

    normalizer = np.array(R + G + B, dtype=np.float32)
    # Avoid division by zero
    normalizer[normalizer == 0] = 1
    r, g, b = (R, G, B) / normalizer

    rExG = 2 * g - (r + b)
    # = -70 * (R-G) - 60(R - B) = 60 * B + 70 * G - 130 * R
    return rExG


def index_NDI(pictureCurrent):

    R, G, B = cv2.split(pictureCurrent)

    normalizer = np.array(R + G + B, dtype=np.float32)
    # Avoid division by zero
    normalizer[normalizer == 0] = 1
    r, g, b = (R, G, B) / normalizer

    rNDI = 128 * (((g-r)/(g+r))+1)
    # = -70 * (R-G) - 60(R - B) = 60 * B + 70 * G - 130 * R
    return rNDI


def index_ExGR(pictureCurrent):

    R, G, B = cv2.split(pictureCurrent)

    normalizer = np.array(R + G + B, dtype=np.float32)
    # Avoid division by zero
    normalizer[normalizer == 0] = 1
    r, g, b = (R, G, B) / normalizer

    rExGR = (2 * g - (r + b)) - (1.3 * r - g)
    # = -70 * (R-G) - 60(R - B) = 60 * B + 70 * G - 130 * R
    return rExGR