#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Deep CNN training for satellite images
"""

import numpy as np
import pandas as pd
import toolbox as tb

__author__ = "Yifei Xue"
__copyright__ = "Copyright 2019, The Xinyu LCLU Project"
__credits__ = ["Yifei Xue"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Yifei Xue"
__email__ = "dla.ml@outlook.com"
__status__ = "Production"

def kaggle_cnn_training():
    x_train, y_train = tb.load_data_and_labels(data_name='../input/X_train_sat6.csv', labels='../input/y_train_sat6.csv')
    x_test, y_test = tb.load_data_and_labels(data_name='../input/X_test_sat6.csv', labels='../input/y_test_sat6.csv')
    pass