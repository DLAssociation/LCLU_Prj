# -*- coding: utf-8 -*-

import numpy as np
import json
import scipy.io as scio
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from osgeo import gdal


def resolve_json(path):
    """ Resolve json data

    :param path: Directory of json file
    :return: json file mapping keys in python format
    """
    file = open(path, "rb")
    file_json = json.load(file)

    return file_json


def read_config(path='config.json', config_name='data_root_directory'):
    """

    :param path: Directory of json file
    :param config_name: the name of configure to fetch
    :return: value of correspongding configure name
    """
    f_json = resolve_json(path)
    if f_json["directories"]["name"] == config_name:
        value = f_json["value"]

    return value


def fetch_sat_mat(sat_name='sat-6-full'):
    """ Fetch SAT4 or SAT6 data from MAT format

    :param sat_name: full name of SAT data. The value of sat_name can be 'sat-4-full' or 'sat-6-full'
    :return: A dict mapping keys

    Load data from MAT-file

    The function fetch_sat_mat loads all variables stored in the MAT-file into a simple Python data structure,
    using only Python’s dict and list objects. Numeric and cell arrays are converted to row-ordered nested lists.
    Arrays are squeezed to eliminate arrays with only one element. The resulting data structure is composed of simple
    types that are compatible with the JSON format.

    Example: Load a MAT-file into a Python data structure:

    data = fetch_sat_mat('datafile.mat')
    The variable data is a dict with the variables and values contained in the MAT-file.

    Save Python data structure to a MAT-file

    Python data can be saved to a MAT-file, with the function savemat. Data has to be structured in the same way as
    for loadmat, i.e. it should be composed of simple data types, like dict, list, str, int and float.

    Example: Save a Python data structure to a MAT-file:

    savemat('datafile.mat', data)
    """

    # Read directory of SAT data
    sat_dir = read_config('config.json', 'data_root_directory') + sat_name + '.mat'

    sat_data = scio.loadmat(sat_dir)
    # sat_train_x = sat_data.get('train_x')
    return sat_data


def get_imgs(sat_name='sat-6-full', column_name='train_x'):
    """

    :param sat_name:
    :param column_name:
    :return:

    column_name can be:
    train_x        --------------    28x28x4x324000 uint8  (containing 324000 training samples of 28x28 images each with 4 channels - R, G, B and NIR)
    train_y        --------------    6x324000       double (containing 6x1 vectors having labels for the 324000 training samples)
    test_x         --------------    28x28x4x81000  uint8  (containing 81000 test samples of 28x28 images each with 4 channels - R, G, B and NIR)
    test_y         --------------    6x81000        double (containing 6x1 vectors having labels for the 81000 test samples)
    annotations    --------------    6x2            cell   (containing the class label annotations for the 6 classes of SAT-6)
    """
    imgs = fetch_sat_mat(sat_name).get(column_name)
    print('The shape of ', column_name, ' is ', imgs.shape)
    return imgs


def csv_writer(csv_name, img_narray):
    csv_dir = read_config('config.json', 'data_root_directory') + csv_name + '.csv'
    pd.DataFrame(img_narray).to_csv(csv_dir, header=None, index=None)


def get_img(imgs, index):
    return imgs[:, :, :, index]


def show_img(img):
    plt.imshow(img, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


def get_annotation(sat_name='sat-6-full'):
    """

    :param sat_name:
    :return:
    """
    note = fetch_sat_mat(sat_name).get('annotations')
    return note


def load_data_and_labels(data_name, labels):
    """
    function to load data and labels
    :param data:
    :param labels:
    :return:
    """
    data_df = pd.read_csv(data_name, header=None)
    X = data_df.values.reshape((-1, 28, 28, 4)).clip(0, 255).astype(np.uint8)
    labels_df = pd.read_csv(labels, header=None)
    Y = labels_df.values.getfield(dtype=np.int8)
    return X, Y

def run_once():
    """
    only run once
    :return:
    """
    img_x = get_imgs('sat-6-full', 'train_x')
    csv_writer('sat-6_train_x', img_x)


def main():
    run_once()
    print('test')


if __name__ == '__main__':
    main()
