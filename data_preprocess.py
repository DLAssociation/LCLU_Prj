#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Deep CNN training for satellite images
"""
from __future__ import print_function
import numpy as np  # linear algebra
import pandas as pd
from keras_applications.vgg16 import VGG16
from keras_applications.vgg19 import VGG19
from PIL import Image
import toolbox as tb
from subprocess import check_output

__author__ = "Yifei Xue"
__copyright__ = "Copyright 2019, The Xinyu LCLU Project"
__credits__ = ["Yifei Xue"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Yifei Xue"
__email__ = "dla.ml@outlook.com"
__status__ = "Production"

print(check_output(["ls", tb.read_config('config.json', 'csv_data_root_directory')]).decode("utf8"))

VGG_LAYER = 16


# get every pixel value from image, and restored it as a numpu array
def getImageData(im_path, start_idx, end_idx):
    imData = []
    for i in range(start_idx, end_idx):
        im = Image.open(im_path + str(i) + '.tif')
        im = np.array(im)
        r = im[:, :, 0]
        g = im[:, :, 1]
        b = im[:, :, 2]
        imData.append([r, g, b])
    return np.array(imData)


def main():
    im_root = tb.read_config('config.json', 'UCMerced_LandUse_directory')

    # Choosing the number of layer from VGG
    if VGG_LAYER == 16:
        extractor = VGG16(weights='imagenet', include_top=False)
    elif VGG_LAYER == 19:
        extractor = VGG19(weights='imagenet', include_top=False)

    # Get ImageData of 1 subsample (1 image every class)
    # 'IMG' is a folder containing image data
    data1 = getImageData('IMG/', 0, 21)
    X = extractor.predict(data1, verbose=0)

    # iterate for all 2100 images (100 times)
    for i in range(1, 100):
        data2 = getImageData('IMG/', i * 21, i * 21 + 21)
        X2 = extractor.predict(data2, verbose=0)
        X = np.concatenate((X, X2), axis=0)
        print(X.shape)  # to clarify only (optional)

    # save the array to external file that will be used for Neural Network
    np.save('Input_NN_VGG' + str(VGG_LAYER) + '.npy', X)

if __name__ == '__main__':
    main()