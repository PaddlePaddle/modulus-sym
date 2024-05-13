#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Nvidia packages of the Norne field

Calls Flow simulator

Geostatistics packages are also provided

@Author: Clement Etienam

"""
print(".........................IMPORT SOME LIBRARIES.....................")
import os
import numpy as np


def is_available():
    """
    Check NVIDIA with nvidia-smi command
    Returning code 0 if no error, it means NVIDIA is installed
    Other codes mean not installed
    """
    code = os.system("nvidia-smi")
    return code


Yet = is_available()
if Yet == 0:
    print("GPU Available with CUDA")

    import cupy as cp
    from numba import cuda

    print(cuda.detect())  # Print the GPU information
    import tensorflow as tf

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.InteractiveSession(config=config)

    clementtt = 0
else:
    print("No GPU Available")
    import numpy as cp

    clementtt = 1

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import MiniBatchKMeans
import os.path
import paddle
from paddle. io import DataLoader

paddle.set_default_dtype("float32")
from joblib import Parallel, delayed
from scipy import interpolate
import multiprocessing
import mpslib as mps
from shutil import rmtree
import numpy.matlib
import re
from sklearn.model_selection import train_test_split
from numpy import linalg as LA
from scipy.spatial.distance import cdist
from pyDOE import lhs
import matplotlib.colors
from matplotlib import cm
import pickle
from copy import copy
import xgboost as xgb
from kneed import KneeLocator
import numpy
from scipy.stats import rankdata, norm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# from PIL import Image
from scipy.fftpack import dct
from paddle.utils.dlpack import to_dlpack
from paddle.utils.dlpack import from_dlpack
import numpy.matlib
from matplotlib import pyplot

# os.environ['KERAS_BACKEND'] = 'tensorflow'
import os.path
import math
import time
import random
from matplotlib.font_manager import FontProperties
import os.path
from datetime import timedelta
import pandas as pd
from collections import namedtuple, defaultdict
from skimage.transform import resize as rzz
from mpl_toolkits.mplot3d import Axes3D
import imp
import h5py
from modulus.sym.hydra import to_absolute_path
import scipy.io as sio
import yaml
import matplotlib
import matplotlib as mpl
import matplotlib.lines as mlines
import os

# import zipfile
import gzip

try:
    import gdown
except:
    gdown = None

import scipy.io
import h5py

from loky import get_reusable_executor
import shutil
import sys
from numpy import *
import scipy.optimize.lbfgsb as lbfgsb
import numpy.linalg
from numpy.linalg import norm
from scipy.fftpack.realtransforms import idct
import numpy.ma as ma
import logging
from FyeldGenerator import generate_field
from imresize import *
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # I have just 1 GPU
from cpuinfo import get_cpu_info

# Prints a json string describing the cpu
s = get_cpu_info()
print("Cpu info")
for k, v in s.items():
    print(f"\t{k}: {v}")
cores = multiprocessing.cpu_count()
import math

logger = logging.getLogger(__name__)
# numpy.random.seed(99)
print(" ")
print(" This computer has %d cores, which will all be utilised in parallel " % cores)
print(" ")
print("......................DEFINE SOME FUNCTIONS.....................")


def fast_gaussian(dimension, Sdev, Corr):
    """
    Generates random vector from distribution satisfying Gaussian
    variogram in 2-d.

    Input:
    - dimension: Dimension of grid
    - Sdev: Standard deviation
    - Corr: Correlation length, in units of block length.
            Corr may be replaced with a vector of length 2 with
            correlation length in x- and y- direction.

    Output:
    - x: Random vector.
    """

    # Ensure dimension is a 1D numpy array
    dimension = np.array(dimension).flatten()
    m = dimension[0]

    if len(dimension) == 1:
        n = m
    elif len(dimension) == 2:
        n = dimension[1]
    else:
        raise ValueError(
            "FastGaussian: Wrong input, dimension should have length at most 2"
        )

    # mxn = m * n

    # Compute variance
    if np.max(np.size(Sdev)) > 1:  # check input
        variance = 1
    else:
        variance = Sdev  # the variance will come out through the kronecker product.

    # Ensure Corr is a 1D numpy array

    # Corr = np.array(Corr).flatten()

    if len(Corr) == 1:
        Corr = np.array([Corr[0], Corr[0]])
    elif len(Corr) > 2:
        raise ValueError("FastGaussian: Wrong input, Corr should have length at most 2")

    # Generate the covariance matrix for one layer
    dist = np.arange(0, m) / Corr[0]
    T = scipy.linalg.toeplitz(dist)

    T = variance * np.exp(-(T**2)) + 1e-10 * np.eye(m)

    # Cholesky decomposition for one layer:
    cholT = np.linalg.cholesky(T)

    # Generate the covariance matrix for the second layer:
    # To save time - use a copy if possible:
    if Corr[0] == Corr[1] and n == m:
        cholT2 = cholT
    else:
        # Same as for the first dimension:
        dist2 = np.arange(0, n) / Corr[1]
        T2 = scipy.linalg.toeplitz(dist2)
        T2 = variance * np.exp(-(T2**2)) + 1e-10 * np.eye(n)
        cholT2 = np.linalg.cholesky(T2)

    # Draw a random variable:
    x = np.random.randn(m * n)

    # Adjust to get the correct covariance matrix,
    # Applying the matrix transformation:
    x = np.dot(cholT.T, np.dot(x.reshape(m, n), cholT2))

    # Reshape back
    x = x.flatten()

    if np.max(np.size(Sdev)) > 1:
        if np.min(np.shape(Sdev)) == 1 and len(Sdev) == len(x):
            x = Sdev * x
        else:
            raise ValueError("FastGaussian: Inconsistent dimension of Sdev")

    return x


def get_shape(t):
    shape = []
    while isinstance(t, tuple):
        shape.append(len(t))
        t = t[0]
    return tuple(shape)


def NorneInitialEnsemble(ensembleSize=100, randomNumber=1.2345e5):
    # N= 2
    # set random generator
    np.random.seed(int(randomNumber))

    # number of ensemble members
    N = ensembleSize

    # set geostatistical parameters and initialize
    norne = NorneGeostat()
    # Assuming norne is a dictionary where properties of the Norne field are stored

    # mask for active gridcells
    A = norne["actnum"]

    # reservoir dimension
    D = norne["dim"]

    # total number of gridcells
    N_F = D[0] * D[1] * D[2]

    # initialize ensemble

    # mean values for poro, log-permx and ntg
    M = [norne["poroMean"], norne["permxLogMean"], 0.6]

    # std for poro, permx and ntg
    S = [norne["poroStd"], norne["permxStd"], norne["ntgStd"]]

    # help variable (layerwise actnum)
    A_L = [A[i : i + D[1] * D[2]] for i in range(0, len(A), D[1] * D[2])]
    A_L = np.array(A_L)

    # mean and std for multflt and multreg
    M_MF = 0.6
    S_MF = norne["multfltStd"]

    # mean correlation lengths (ranges)
    C = [norne["poroRange"], norne["permxRange"], norne["ntgRange"]]

    # std for correlation lengths

    C_S = 2

    # Poro / Permx correlation
    R1 = norne["poroPermxCorr"]

    ensembleperm = np.zeros((N_F, N))
    ensemblefault = np.zeros((53, N))
    ensembleporo = np.zeros((N_F, N))

    indices = np.where(A == 1)
    for i in range(N):

        # multz
        A_MZ = A_L[:, [0, 7, 10, 11, 14, 17]]  # Adjusted indexing to 0-based
        A_MZ = A_MZ.flatten()

        X = M_MF + S_MF * np.random.randn(53)
        ensemblefault[:, i] = X

        # poro
        C = np.array(C)
        X1 = gaussian_with_variable_parameters(D, np.zeros(N_F), 1, C[0], C_S)[0]
        X1 = X1.reshape(-1, 1)

        ensembleporo[indices, i] = (M[0] + S[0] * X1[indices]).ravel()

        # permx
        X2 = gaussian_with_variable_parameters(D, np.zeros(N_F), 1, C[1], C_S)[0]
        X2 = X2.reshape(-1, 1)
        X = R1 * X1 + np.sqrt(1 - R1**2) * X2

        indices = np.where(A == 1)
        ensembleperm[indices, i] = (M[1] + S[1] * X[indices]).ravel()  #
    return ensembleperm, ensembleporo, ensemblefault


def gaussian_with_variable_parameters(
    field_dim, mean_value, sdev, mean_corr_length, std_corr_length
):
    """
    Setup a Gaussian field with correlation length drawn from a normal distribution.
    The horizontal layers are generated independently.

    Parameters:
    - field_dim        : Tuple. Dimension of the field.
    - mean_value       : Numpy array. The mean value of the field.
    - sdev             : Numpy array or float. Standard deviation of the field.
    - mean_corr_length : Float. Mean correlation length.
    - std_corr_length  : Float. Standard deviation of the correlation length.

    Returns:
    - x                : Numpy array. The generated field.
    - corr_length      : Numpy array or float. The drawn correlation length.
    """

    corr_length = add_gnoise(mean_corr_length, std_corr_length, 1)

    if len(field_dim) < 3:
        x = mean_value + fast_gaussian(field_dim, sdev, corr_length)
    else:
        layer_dim = np.prod(field_dim[:2])
        x = np.copy(mean_value)

        if np.isscalar(sdev):
            for i in range(field_dim[2]):
                idx_range = slice(i * layer_dim, (i + 1) * layer_dim)
                x[idx_range] = mean_value[idx_range] + fast_gaussian(
                    field_dim[:2], sdev, corr_length
                )
                # Generate new correlation length for the next layer
                corr_length = add_gnoise(mean_corr_length, std_corr_length, 1)
        else:
            for i in range(field_dim[2]):
                idx_range = slice(i * layer_dim, (i + 1) * layer_dim)
                x[idx_range] = mean_value[idx_range] + fast_gaussian(
                    field_dim[:2], sdev[idx_range], corr_length
                )
                # Generate new correlation length for the next layer
                corr_length = add_gnoise(mean_corr_length, std_corr_length, 1)

    return x, corr_length


# Placeholder functions for the ones referred to in the MATLAB code:


def add_gnoise(Ytrue, SIGMA, SQ=None):
    """
    Add noise, normally distributed, with covariance given by SIGMA.

    Parameters:
    - Ytrue    Original signal.
    - SIGMA    Specification of covariance matrix of noise. May be entered as scalar, vector or full matrix.
               If SIGMA is a vector, then it is interpreted as the covariance matrix is diag(SIGMA).
    - SQ       If provided, determine whether SIGMA or SIGMA*SIGMA' is used as the covariance matrix.
               If the square root of the covariance matrix has already been calculated previously, work may
               be saved by setting SQ to 1.

    Returns:
    - Y        Signal with noise added.
    - RTSIGMA  The square root of SIGMA; RTSIGMA @ RTSIGMA.T = SIGMA.
               (Helpful if it is cumbersome to compute).
    """

    try:
        if SQ is not None and SQ == 1:
            # Use SIGMA*SIGMA' as covariance matrix
            RTSIGMA = SIGMA
            if np.isscalar(SIGMA) or np.ndim(SIGMA) == 1:
                # SIGMA is a scalar or vector
                error = RTSIGMA * np.random.randn(1)
            else:
                error = RTSIGMA @ np.random.randn(RTSIGMA.shape[1], 1)
        else:
            # Use SIGMA as covariance matrix
            if np.isscalar(SIGMA) or np.ndim(SIGMA) == 1:
                # SIGMA is entered as a scalar or a vector
                RTSIGMA = np.sqrt(SIGMA)
                error = RTSIGMA * np.random.randn(*Ytrue.shape)
            else:
                # The matrix must be transposed.
                try:
                    RTSIGMA = np.linalg.cholesky(SIGMA).T
                except np.linalg.LinAlgError:
                    print("Problem with Cholesky factorization")
                    RTSIGMA = np.sqrtm(SIGMA).real
                    print("Finally - we got a square root!")

                error = RTSIGMA @ np.random.randn(*Ytrue.shape)

        # Add the noise:
        Y = Ytrue + error.flatten()

    except Exception as e:
        print("Error in AddGnoise")
        raise e

    return Y, RTSIGMA


def adjust_variable_within_bounds(variable, lowerbound=None, upperbound=None):
    """
    Adjust variable such that lowerbound <= variable(i) <= upperbound.

    Parameters:
    - variable: numpy array. Variables (or an ensemble of variable samples) to be checked and truncated (if necessary).
    - lowerbound: Scalar or numpy array. Lower bound(s) for the variables to be checked.
    - upperbound: Scalar or numpy array. Upper bound(s) for the variables to be checked.

    Returns:
    - variable: Variables after check/truncation.
    - n: Number of truncations.
    """

    if lowerbound is None and upperbound is None:
        raise ValueError("At least one of lowerbound or upperbound must be provided.")

    n = 0
    ne = variable.shape[1]

    if lowerbound is not None:
        if np.isscalar(lowerbound):
            n += np.sum(variable < lowerbound)
            variable[variable < lowerbound] = lowerbound
        else:
            lowerbound_repeated = np.tile(lowerbound.reshape(-1, 1), (1, ne))
            n += np.sum(variable < lowerbound_repeated)
            variable[variable < lowerbound_repeated] = lowerbound_repeated[
                variable < lowerbound_repeated
            ]

    if upperbound is not None:
        if np.isscalar(upperbound):
            n += np.sum(variable > upperbound)
            variable[variable > upperbound] = upperbound
        else:
            upperbound_repeated = np.tile(upperbound.reshape(-1, 1), (1, ne))
            n += np.sum(variable > upperbound_repeated)
            variable[variable > upperbound_repeated] = upperbound_repeated[
                variable > upperbound_repeated
            ]

    return variable, n


def read_until_line(file_path, line_num, skip=0, sep="\s+", header=None):
    """
    Reads a CSV file up to a specific line and returns it as a numpy array.

    Parameters:
    - file_path (str): Path to the CSV file.
    - line_num (int): The line number up to which the file should be read.
    - skip (int, optional): Number of rows to skip at the start. Default is 0.
    - sep (str, optional): Delimiter to use. Default is '\s+' (whitespace).
    - header (str or None, optional): Row number to use as the column names. Default is None.

    Returns:
    - np.array: A numpy array containing the read data.
    """

    # Calculate number of rows to read
    nrows_to_read = line_num - skip

    # Read the file using pandas
    df = pd.read_csv(
        file_path, skiprows=skip, nrows=nrows_to_read, sep=sep, header=header
    )

    # Convert DataFrame to numpy array and return
    return df.values


def NorneGeostat():
    norne = {}

    dim = np.array([46, 112, 22])
    ldim = dim[0] * dim[1]
    norne["dim"] = dim

    # actnum
    # act = pd.read_csv('../Norne_Initial_ensemble/ACTNUM_0704.prop', skiprows=8,nrows = 2472, sep='\s+', header=None)
    act = read_until_line("../Necessaryy/ACTNUM_0704.prop", line_num=2465, skip=1)
    act = act.T
    act = np.reshape(act, (-1,), "F")
    norne["actnum"] = act

    # porosity
    meanv = np.zeros(dim[2])
    stdv = np.zeros(dim[2])
    file_path = "../Necessaryy/porosity.dat"
    p = read_until_line(file_path, line_num=113345, skip=1)
    p = p[act != 0]

    for nr in range(int(dim[2])):

        index_start = ldim * nr
        index_end = ldim * (nr + 1)
        values_range_start = int(np.sum(act[:index_start]))
        values_range_end = int(np.sum(act[:index_end]))
        values = p[values_range_start:values_range_end]

        meanv[nr] = np.mean(values)
        stdv[nr] = np.std(values)

    norne["poroMean"] = p
    norne["poroLayerMean"] = meanv
    norne["poroLayerStd"] = stdv
    norne["poroStd"] = 0.05
    norne["poroLB"] = 0.1
    norne["poroUB"] = 0.4
    norne["poroRange"] = 26

    # permeability

    # Permeability

    k = read_until_line("../Necessaryy/permx.dat", line_num=113345, skip=1)
    k = np.log(k)
    k = k[act != 0]

    meanv = np.zeros(dim[2])
    stdv = np.zeros(dim[2])

    for nr in range(int(dim[2])):

        index_start = ldim * nr
        index_end = ldim * (nr + 1)
        values_range_start = int(np.sum(act[:index_start]))
        values_range_end = int(np.sum(act[:index_end]))
        values = k[values_range_start:values_range_end]

        meanv[nr] = np.mean(values)
        stdv[nr] = np.std(values)

    norne["permxLogMean"] = k
    norne["permxLayerLnMean"] = meanv
    norne["permxLayerStd"] = stdv
    norne["permxStd"] = 1
    norne["permxLB"] = 0.1
    norne["permxUB"] = 10
    norne["permxRange"] = 26

    # Correlation between layers

    corr_with_next_layer = np.zeros(dim[2] - 1)

    for nr in range(dim[2] - 1):
        index_start = ldim * nr
        index_end = ldim * (nr + 1)

        index2_start = ldim * (nr + 1)
        index2_end = ldim * (nr + 2)

        act_layer1 = act[index_start:index_end]
        act_layer2 = act[index2_start:index2_end]

        active = act_layer1 * act_layer2

        values1_range_start = int(np.sum(act[:index_start]))
        values1_range_end = int(np.sum(act[:index_end]))
        values1 = np.concatenate(
            (
                k[values1_range_start:values1_range_end],
                p[values1_range_start:values1_range_end],
            )
        )

        values2_range_start = int(np.sum(act[:index2_start]))
        values2_range_end = int(np.sum(act[:index2_end]))
        values2 = np.concatenate(
            (
                k[values2_range_start:values2_range_end],
                p[values2_range_start:values2_range_end],
            )
        )

        v1 = np.concatenate((act_layer1, act_layer1))
        v1[v1 == 1] = values1.flatten()

        v2 = np.concatenate((act_layer2, act_layer2))
        v2[v2 == 1] = values2.flatten()

        active_full = np.concatenate((active, active))
        co = np.corrcoef(v1[active_full == 1], v2[active_full == 1])

        corr_with_next_layer[nr] = co[0, 1]

    norne["corr_with_next_layer"] = corr_with_next_layer.T

    # Correlation between porosity and permeability
    norne["poroPermxCorr"] = 0.7

    norne["poroNtgCorr"] = 0.6
    norne["ntgStd"] = 0.1
    norne["ntgLB"] = 0.01
    norne["ntgUB"] = 1
    norne["ntgRange"] = 26

    # rel-perm end-point scaling
    norne["krwMean"] = 1.15
    norne["krwLB"] = 0.8
    norne["krwUB"] = 1.5
    norne["krgMean"] = 0.9
    norne["krgLB"] = 0.8
    norne["krgUB"] = 1

    # oil-water contact
    norne["owcMean"] = np.array([2692.0, 2585.5, 2618.0, 2400.0, 2693.3])
    norne["owcLB"] = norne["owcMean"] - 10
    norne["owcUB"] = norne["owcMean"] + 10

    # region multipliers
    norne["multregtLogMean"] = np.log10(np.array([0.0008, 0.1, 0.05]))
    norne["multregtStd"] = 0.5
    norne["multregtLB"] = -5
    norne["multregtUB"] = 0

    # z-multipliers
    z_means = [-2, -1.3, -2, -2, -2, -2]
    z_stds = [0.5, 0.5, 0.5, 0.5, 1, 1]
    for i, (mean_, std_) in enumerate(zip(z_means, z_stds), start=1):
        norne[f"z{i}Mean"] = mean_
        norne[f"z{i}Std"] = std_
    norne["zLB"] = -4
    norne["zUB"] = 0
    norne["multzRange"] = 26

    # fault multipliers
    norne["multfltStd"] = 0.5
    norne["multfltLB"] = -5
    norne["multfltUB"] = 2

    return norne


def Reinvent(matt):
    nx, ny, nz = matt.shape[1], matt.shape[2], matt.shape[0]
    dess = np.zeros((nx, ny, nz))
    for i in range(nz):
        dess[:, :, i] = matt[i, :, :]
    return dess


def Add_marker(plt, XX, YY, locc):
    """
    Function to add marker to given coordinates on a matplotlib plot

    less
    Copy code
    Parameters:
        plt: a matplotlib.pyplot object to add the markers to
        XX: a numpy array of X coordinates
        YY: a numpy array of Y coordinates
        locc: a numpy array of locations where markers need to be added

    Return:
        None
    """
    # iterate through each location
    for i in range(locc.shape[0]):
        a = locc[i, :]
        xloc = int(a[0])
        yloc = int(a[1])

        # if the location type is 2, add an upward pointing marker
        if a[2] == 2:
            plt.scatter(
                XX.T[xloc - 1, yloc - 1] + 0.5,
                YY.T[xloc - 1, yloc - 1] + 0.5,
                s=100,
                marker="^",
                color="white",
            )
        # otherwise, add a downward pointing marker
        else:
            plt.scatter(
                XX.T[xloc - 1, yloc - 1] + 0.5,
                YY.T[xloc - 1, yloc - 1] + 0.5,
                s=100,
                marker="v",
                color="white",
            )


def Plot_RSM_percentile(pertoutt, True_mat, timezz):

    columns = [
        "B_1BH",
        "B_1H",
        "B_2H",
        "B_3H",
        "B_4BH",
        "B_4DH",
        "B_4H",
        "D_1CH",
        "D_1H",
        "D_2H",
        "D_3AH",
        "D_3BH",
        "D_4AH",
        "D_4H",
        "E_1H",
        "E_2AH",
        "E_2H",
        "E_3AH",
        "E_3CH",
        "E_3H",
        "E_4AH",
        "K_3H",
    ]

    # timezz = True_mat[:,0].reshape(-1,1)

    P10 = pertoutt

    plt.figure(figsize=(40, 40))

    for k in range(22):
        plt.subplot(5, 5, int(k + 1))
        plt.plot(timezz, True_mat[:, k], color="red", lw="2", label="Flow")
        plt.plot(timezz, P10[:, k], color="blue", lw="2", label="Modulus")
        plt.xlabel("Time (days)", fontsize=13, fontweight="bold")
        plt.ylabel("$Q_{oil}(bbl/day)$", fontsize=13, fontweight="bold")
        # plt.ylim((0,25000))
        plt.title(columns[k], fontsize=13, fontweight="bold")
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        plt.legend()

    # os.chdir('RESULTS')
    plt.suptitle(
        "NORNE - Oil Production ($Q_{oil}(bbl/day)$)", fontsize=16, fontweight="bold"
    )
    plt.savefig(
        "Oil.png"
    )  # save as png                                  # preventing the figures from showing
    # os.chdir(oldfolder)
    plt.clf()
    plt.close()

    plt.figure(figsize=(40, 40))

    for k in range(22):
        plt.subplot(5, 5, int(k + 1))
        plt.plot(timezz, True_mat[:, k + 22], color="red", lw="2", label="Flow")
        plt.plot(timezz, P10[:, k + 22], color="blue", lw="2", label="Modulus")
        plt.xlabel("Time (days)", fontsize=13, fontweight="bold")
        plt.ylabel("$Q_{water}(bbl/day)$", fontsize=13, fontweight="bold")
        # plt.ylim((0,25000))
        plt.title(columns[k], fontsize=13, fontweight="bold")
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        plt.legend()

    # os.chdir('RESULTS')
    plt.suptitle(
        "NORNE - Water Production ($Q_{water}(bbl/day)$)",
        fontsize=16,
        fontweight="bold",
    )
    plt.savefig(
        "Water.png"
    )  # save as png                                  # preventing the figures from showing
    # os.chdir(oldfolder)
    plt.clf()
    plt.close()

    plt.figure(figsize=(40, 40))

    for k in range(22):
        plt.subplot(5, 5, int(k + 1))
        plt.plot(timezz, True_mat[:, k + 44], color="red", lw="2", label="Flow")
        plt.plot(timezz, P10[:, k + 44], color="blue", lw="2", label="Modulus")
        plt.xlabel("Time (days)", fontsize=13, fontweight="bold")
        plt.ylabel("$Q_{gas}(scf/day)$", fontsize=13, fontweight="bold")
        # plt.ylim((0,25000))
        plt.title(columns[k], fontsize=13, fontweight="bold")
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        plt.legend()

    # os.chdir('RESULTS')
    plt.suptitle(
        "NORNE - Gas Production ($Q_{gas}(scf/day)$)", fontsize=16, fontweight="bold"
    )

    plt.savefig(
        "Gas.png"
    )  # save as png                                  # preventing the figures from showing
    # os.chdir(oldfolder)
    plt.clf()
    plt.close()


def MyLossClement(a, b):

    loss = paddle.sum(paddle.abs(a - b) / a.shape[0])

    # loss = ((a-b)**2).mean()
    return loss


# Geostatistics module
def intial_ensemble(Nx, Ny, Nz, N, permx):
    """
    Geostatistics module
    Function to generate an initial ensemble of permeability fields using Multiple-Point Statistics (MPS)
    Parameters:
        Nx: an integer representing the number of grid cells in the x-direction
        Ny: an integer representing the number of grid cells in the y-direction
        Nz: an integer representing the number of grid cells in the z-direction
        N: an integer representing the number of realizations in the ensemble
        permx: a numpy array representing the permeability field TI

    Return:
        ensemble: a numpy array representing the ensemble of permeability fields
    """

    # import MPSlib
    O = mps.mpslib()

    # set the MPS method to 'mps_snesim_tree'
    O = mps.mpslib(method="mps_snesim_tree")

    # set the number of realizations to N
    O.par["n_real"] = N

    # set the permeability field TI
    k = permx
    kjenn = k
    O.ti = kjenn

    # set the simulation grid size
    O.par["simulation_grid_size"] = (Ny, Nx, Nz)

    # run MPS simulation in parallel
    O.run_parallel()

    # get the ensemble of realizations
    ensemble = O.sim

    # reformat the ensemble
    ens = []
    for kk in range(N):
        temp = np.reshape(ensemble[kk], (-1, 1), "F")
        ens.append(temp)
    ensemble = np.hstack(ens)

    # remove temporary files generated during MPS simulation
    from glob import glob

    for f3 in glob("thread*"):
        rmtree(f3)

    for f4 in glob("*mps_snesim_tree_*"):
        os.remove(f4)

    for f4 in glob("*ti_thread_*"):
        os.remove(f4)

    return ensemble


def initial_ensemble_gaussian(Nx, Ny, Nz, N, minn, maxx):
    """
    Function to generate an initial ensemble of permeability fields using Gaussian distribution
    Parameters:
        Nx: an integer representing the number of grid cells in the x-direction
        Ny: an integer representing the number of grid cells in the y-direction
        Nz: an integer representing the number of grid cells in the z-direction
        N: an integer representing the number of realizations in the ensemble
        minn: a float representing the minimum value of the permeability field
        maxx: a float representing the maximum value of the permeability field

    Return:
        fensemble: a numpy array representing the ensemble of permeability fields
    """

    shape = (Nx, Ny)
    distrib = "gaussian"

    fensemble = np.zeros((Nx * Ny * Nz, N))

    for k in range(N):
        fout = []

        # generate a 3D field
        for j in range(Nz):
            field = generate_field(distrib, Pkgen(3), shape)
            field = imresize(field, output_shape=shape)
            foo = np.reshape(field, (-1, 1), "F")
            fout.append(foo)

        fout = np.vstack(fout)

        # scale the field to the desired range
        clfy = MinMaxScaler(feature_range=(minn, maxx))
        (clfy.fit(fout))
        fout = clfy.transform(fout)

        fensemble[:, k] = np.ravel(fout)

    return fensemble


def Pkgen(n):
    def Pk(k):
        return np.power(k, -n)

    return Pk


# Draw samples from a normal distribution
def distrib(shape):
    a = np.random.normal(loc=0, scale=1, size=shape)
    b = np.random.normal(loc=0, scale=1, size=shape)
    return a + 1j * b


# Points generation


def test_points_gen(n_test, nder, interval=(-1.0, 1.0), distrib="random", **kwargs):
    return {
        "random": lambda n_test, nder: (interval[1] - interval[0])
        * np.random.rand(n_test, nder)
        + interval[0],
        "lhs": lambda n_test, nder: (interval[1] - interval[0])
        * lhs(nder, samples=n_test, **kwargs)
        + interval[0],
    }[distrib.lower()](n_test, nder)


def getoptimumk(X):
    distortions = []
    Kss = range(1, 10)

    for k in Kss:
        kmeanModel = MiniBatchKMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(
            sum(np.min(cdist(X, kmeanModel.cluster_centers_, "euclidean"), axis=1))
            / X.shape[0]
        )

    myarray = np.array(distortions)

    knn = KneeLocator(
        Kss, myarray, curve="convex", direction="decreasing", interp_method="interp1d"
    )
    kuse = knn.knee

    # Plot the elbow
    plt.figure(figsize=(10, 10))
    plt.plot(Kss, distortions, "bx-")
    plt.xlabel("cluster size")
    plt.ylabel("Distortion")
    plt.title("optimal n_clusters for machine")
    plt.savefig("machine_elbow.png")
    plt.clf()
    return kuse


class LpLoss(object):
    """
    loss function with rel/abs Lp loss
    """

    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * paddle.norm(
            x.reshape([num_examples, -1]) - y.reshape([num_examples, -1]), self.p, 1
        )

        if self.reduction:
            if self.size_average:
                return paddle.mean(all_norms)
            else:
                return paddle.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = paddle.norm(
            x.reshape([num_examples, -1]) - y.reshape([num_examples, -1]), self.p, 1
        )
        y_norms = paddle.norm(y.reshape([num_examples, -1]), self.p, 1)

        if self.reduction:
            if self.size_average:
                return paddle.mean(diff_norms / y_norms)
            else:
                return paddle.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def round_array_to_4dp(arr):
    """
    Rounds the values of an input array to 4 decimal places.

    Args:
    - arr (numpy.ndarray or array-like): The input array.

    Returns:
    - numpy.ndarray: Array with values rounded to 4 decimal places.
    """
    try:
        arr = np.asarray(arr)  # Convert input to a numpy array if it's not already
        return np.around(arr, 4)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None  # You can choose to return None or handle the error differently


def H(y, t0=0):
    """
    Step fn with step at t0
    """
    # h = np.zeros_like(y)
    # args = tuple([slice(0,y.shape[i]) for i in y.ndim])


def smoothn(
    y,
    nS0=10,
    axis=None,
    smoothOrder=2.0,
    sd=None,
    verbose=False,
    s0=None,
    z0=None,
    isrobust=False,
    W=None,
    s=None,
    MaxIter=100,
    TolZ=1e-3,
    weightstr="bisquare",
):

    if type(y) == ma.core.MaskedArray:  # masked array
        # is_masked = True
        mask = y.mask
        y = np.array(y)
        y[mask] = 0.0
        if np.any(W != None):
            W = np.array(W)
            W[mask] = 0.0
        if np.any(sd != None):
            W = np.array(1.0 / sd**2)
            W[mask] = 0.0
            sd = None
        y[mask] = np.nan

    if np.any(sd != None):
        sd_ = np.array(sd)
        mask = sd > 0.0
        W = np.zeros_like(sd_)
        W[mask] = 1.0 / sd_[mask] ** 2
        sd = None

    if np.any(W != None):
        W = W / W.max()

    sizy = y.shape

    # sort axis
    if axis == None:
        axis = tuple(np.arange(y.ndim))

    noe = y.size  # number of elements
    if noe < 2:
        z = y
        exitflag = 0
        Wtot = 0
        return z, s, exitflag, Wtot
    # ---
    # Smoothness parameter and weights
    # if s != None:
    #  s = []
    if np.all(W == None):
        W = np.ones(sizy)

    # if z0 == None:
    #  z0 = y.copy()

    # ---
    # "Weighting function" criterion
    weightstr = weightstr.lower()
    # ---
    # Weights. Zero weights are assigned to not finite values (Inf or NaN),
    # (Inf/NaN values = missing data).
    IsFinite = np.array(np.isfinite(y)).astype(bool)
    nof = IsFinite.sum()  # number of finite elements
    W = W * IsFinite
    if any(W < 0):
        raise RuntimeError("smoothn:NegativeWeights", "Weights must all be >=0")
    else:
        # W = W/np.max(W)
        pass
    # ---
    # Weighted or missing data?
    isweighted = any(W != 1)
    # ---
    # Robust smoothing?
    # isrobust
    # ---
    # Automatic smoothing?
    isauto = not s
    # ---
    # DCTN and IDCTN are required
    try:
        from scipy.fftpack.realtransforms import dct, idct
    except:
        z = y
        exitflag = -1
        Wtot = 0
        return z, s, exitflag, Wtot

    ## Creation of the Lambda tensor
    # ---
    # Lambda contains the eingenvalues of the difference matrix used in this
    # penalized least squares process.
    axis = tuple(np.array(axis).flatten())
    d = y.ndim
    Lambda = np.zeros(sizy)
    for i in axis:
        # create a 1 x d array (so e.g. [1,1] for a 2D case
        siz0 = np.ones((1, y.ndim))[0].astype(int)
        siz0[i] = sizy[i]
        # cos(pi*(reshape(1:sizy(i),siz0)-1)/sizy(i)))
        # (arange(1,sizy[i]+1).reshape(siz0) - 1.)/sizy[i]
        Lambda = Lambda + (
            np.cos(np.pi * (np.arange(1, sizy[i] + 1) - 1.0) / sizy[i]).reshape(siz0)
        )
        # else:
        #  Lambda = Lambda + siz0
    Lambda = -2.0 * (len(axis) - Lambda)
    if not isauto:
        Gamma = 1.0 / (1 + (s * abs(Lambda)) ** smoothOrder)

    ## Upper and lower bound for the smoothness parameter
    # The average leverage (h) is by definition in [0 1]. Weak smoothing occurs
    # if h is close to 1, while over-smoothing appears when h is near 0. Upper
    # and lower bounds for h are given to avoid under- or over-smoothing. See
    # equation relating h to the smoothness parameter (Equation #12 in the
    # referenced CSDA paper).
    N = sum(np.array(sizy) != 1)
    # tensor rank of the y-array
    hMin = 1e-6
    hMax = 0.99
    # (h/n)**2 = (1 + a)/( 2 a)
    # a = 1/(2 (h/n)**2 -1)
    # where a = sqrt(1 + 16 s)
    # (a**2 -1)/16
    try:
        sMinBnd = np.sqrt(
            (
                ((1 + np.sqrt(1 + 8 * hMax ** (2.0 / N))) / 4.0 / hMax ** (2.0 / N))
                ** 2
                - 1
            )
            / 16.0
        )
        sMaxBnd = np.sqrt(
            (
                ((1 + np.sqrt(1 + 8 * hMin ** (2.0 / N))) / 4.0 / hMin ** (2.0 / N))
                ** 2
                - 1
            )
            / 16.0
        )
    except:
        sMinBnd = None
        sMaxBnd = None
    ## Initialize before iterating
    # ---
    Wtot = W
    # --- Initial conditions for z
    if isweighted:
        # --- With weighted/missing data
        # An initial guess is provided to ensure faster convergence. For that
        # purpose, a nearest neighbor interpolation followed by a coarse
        # smoothing are performed.
        # ---
        if z0 != None:  # an initial guess (z0) has been provided
            z = z0
        else:
            z = y  # InitialGuess(y,IsFinite);
            z[~IsFinite] = 0.0
    else:
        z = np.zeros(sizy)
    # ---
    z0 = z
    y[~IsFinite] = 0
    # arbitrary values for missing y-data
    # ---
    tol = 1.0
    RobustIterativeProcess = True
    RobustStep = 1
    nit = 0
    # --- Error on p. Smoothness parameter s = 10^p
    errp = 0.1
    # opt = optimset('TolX',errp);
    # --- Relaxation factor RF: to speedup convergence
    RF = 1 + 0.75 * isweighted
    # ??
    ## Main iterative process
    # ---
    if isauto:
        try:
            xpost = np.array([(0.9 * np.log10(sMinBnd) + np.log10(sMaxBnd) * 0.1)])
        except:
            np.array([100.0])
    else:
        xpost = np.array([np.log10(s)])
    while RobustIterativeProcess:
        # --- "amount" of weights (see the function GCVscore)
        aow = sum(Wtot) / noe
        # 0 < aow <= 1
        # ---
        while tol > TolZ and nit < MaxIter:
            if verbose:
                print("tol", tol, "nit", nit)
            nit = nit + 1
            DCTy = dctND(Wtot * (y - z) + z, f=dct)
            if isauto and not np.remainder(np.log2(nit), 1):
                # ---
                # The generalized cross-validation (GCV) method is used.
                # We seek the smoothing parameter s that minimizes the GCV
                # score i.e. s = Argmin(GCVscore).
                # Because this process is time-consuming, it is performed from
                # time to time (when nit is a power of 2)
                # ---
                # errp in here somewhere

                # xpost,f,d = lbfgsb.fmin_l_bfgs_b(gcv,xpost,fprime=None,factr=10.,\
                #   approx_grad=True,bounds=[(log10(sMinBnd),log10(sMaxBnd))],\
                #   args=(Lambda,aow,DCTy,IsFinite,Wtot,y,nof,noe))

                # if we have no clue what value of s to use, better span the
                # possible range to get a reasonable starting point ...
                # only need to do it once though. nS0 is teh number of samples used
                if not s0:
                    ss = np.arange(nS0) * (1.0 / (nS0 - 1.0)) * (
                        np.log10(sMaxBnd) - np.log10(sMinBnd)
                    ) + np.log10(sMinBnd)
                    g = np.zeros_like(ss)
                    for i, p in enumerate(ss):
                        g[i] = gcv(
                            p,
                            Lambda,
                            aow,
                            DCTy,
                            IsFinite,
                            Wtot,
                            y,
                            nof,
                            noe,
                            smoothOrder,
                        )
                        # print 10**p,g[i]
                    xpost = [ss[g == g.min()]]
                    # print '==============='
                    # print nit,tol,g.min(),xpost[0],s
                    # print '==============='
                else:
                    xpost = [s0]
                xpost, f, d = lbfgsb.fmin_l_bfgs_b(
                    gcv,
                    xpost,
                    fprime=None,
                    factr=1e7,
                    approx_grad=True,
                    bounds=[(np.log10(sMinBnd), np.log10(sMaxBnd))],
                    args=(Lambda, aow, DCTy, IsFinite, Wtot, y, nof, noe, smoothOrder),
                )
            s = 10 ** xpost[0]
            # update the value we use for the initial s estimate
            s0 = xpost[0]

            Gamma = 1.0 / (1 + (s * abs(Lambda)) ** smoothOrder)

            z = RF * dctND(Gamma * DCTy, f=idct) + (1 - RF) * z
            # if no weighted/missing data => tol=0 (no iteration)
            tol = isweighted * norm(z0 - z) / norm(z)

            z0 = z
            # re-initialization
        exitflag = nit < MaxIter

        if isrobust:  # -- Robust Smoothing: iteratively re-weighted process
            # --- average leverage
            h = np.sqrt(1 + 16.0 * s)
            h = np.sqrt(1 + h) / np.sqrt(2) / h
            h = h**N
            # --- take robust weights into account
            Wtot = W * RobustWeights(y - z, IsFinite, h, weightstr)
            # --- re-initialize for another iterative weighted process
            isweighted = True
            tol = 1
            nit = 0
            # ---
            RobustStep = RobustStep + 1
            RobustIterativeProcess = RobustStep < 3
            # 3 robust steps are enough.
        else:
            RobustIterativeProcess = False
            # stop the whole process

    ## Warning messages
    # ---
    if isauto:
        if abs(np.log10(s) - np.log10(sMinBnd)) < errp:
            warning(
                "MATLAB:smoothn:SLowerBound",
                [
                    "s = %.3f " % (s)
                    + ": the lower bound for s "
                    + "has been reached. Put s as an input variable if required."
                ],
            )
        elif abs(np.log10(s) - np.log10(sMaxBnd)) < errp:
            warning(
                "MATLAB:smoothn:SUpperBound",
                [
                    "s = %.3f " % (s)
                    + ": the upper bound for s "
                    + "has been reached. Put s as an input variable if required."
                ],
            )
    return z, s, exitflag, Wtot


def warning(s1, s2):
    print(s1)
    print(s2[0])


## GCV score
# ---
# function GCVscore = gcv(p)
def gcv(p, Lambda, aow, DCTy, IsFinite, Wtot, y, nof, noe, smoothOrder):
    # Search the smoothing parameter s that minimizes the GCV score
    # ---
    s = 10**p
    Gamma = 1.0 / (1 + (s * abs(Lambda)) ** smoothOrder)
    # --- RSS = Residual sum-of-squares
    if aow > 0.9:  # aow = 1 means that all of the data are equally weighted
        # very much faster: does not require any inverse DCT
        RSS = norm(DCTy * (Gamma - 1.0)) ** 2
    else:
        # take account of the weights to calculate RSS:
        yhat = dctND(Gamma * DCTy, f=idct)
        RSS = norm(np.sqrt(Wtot[IsFinite]) * (y[IsFinite] - yhat[IsFinite])) ** 2
    # ---
    TrH = sum(Gamma)
    GCVscore = RSS / float(nof) / (1.0 - TrH / float(noe)) ** 2
    return GCVscore


## Robust weights
# function W = RobustWeights(r,I,h,wstr)
def RobustWeights(r, I, h, wstr):
    # weights for robust smoothing.
    MAD = np.median(abs(r[I] - np.median(r[I])))
    # median absolute deviation
    u = abs(r / (1.4826 * MAD) / np.sqrt(1 - h))
    # studentized residuals
    if wstr == "cauchy":
        c = 2.385
        W = 1.0 / (1 + (u / c) ** 2)
        # Cauchy weights
    elif wstr == "talworth":
        c = 2.795
        W = u < c
        # Talworth weights
    else:
        c = 4.685
        W = (1 - (u / c) ** 2) ** 2.0 * ((u / c) < 1)
        # bisquare weights

    W[np.isnan(W)] = 0
    return W


## Initial Guess with weighted/missing data
# function z = InitialGuess(y,I)
def InitialGuess(y, I):
    # -- nearest neighbor interpolation (in case of missing values)
    if any(~I):
        try:
            from scipy.ndimage.morphology import distance_transform_edt

            # if license('test','image_toolbox')
            # [z,L] = bwdist(I);
            L = distance_transform_edt(1 - I)
            z = y
            z[~I] = y[L[~I]]
        except:
            # If BWDIST does not exist, NaN values are all replaced with the
            # same scalar. The initial guess is not optimal and a warning
            # message thus appears.
            z = y
            z[~I] = np.mean(y[I])
    else:
        z = y
    # coarse fast smoothing
    z = dctND(z, f=dct)
    k = np.array(z.shape)
    m = np.ceil(k / 10) + 1
    d = []
    for i in np.xrange(len(k)):
        d.append(np.arange(m[i], k[i]))
    d = np.array(d).astype(int)
    z[d] = 0.0
    z = dctND(z, f=idct)
    return z


def dctND(data, f=dct):
    nd = len(data.shape)
    if nd == 1:
        return f(data, norm="ortho", type=2)
    elif nd == 2:
        return f(f(data, norm="ortho", type=2).T, norm="ortho", type=2).T
    elif nd == 3:
        return f(
            f(f(data, norm="ortho", type=2, axis=0), norm="ortho", type=2, axis=1),
            norm="ortho",
            type=2,
            axis=2,
        )
    elif nd == 4:
        return f(
            f(
                f(f(data, norm="ortho", type=2, axis=0), norm="ortho", type=2, axis=1),
                norm="ortho",
                type=2,
                axis=2,
            ),
            norm="ortho",
            type=2,
            axis=3,
        )


def peaks(n):
    """
    Mimic basic of matlab peaks fn
    """
    xp = np.arange(n)
    [x, y] = np.meshgrid(xp, xp)
    z = np.zeros_like(x).astype(float)
    for i in np.xrange(n / 5):
        x0 = random() * n
        y0 = random() * n
        sdx = random() * n / 4.0
        sdy = sdx
        c = random() * 2 - 1.0
        f = np.exp(
            -(((x - x0) / sdx) ** 2)
            - ((y - y0) / sdy) ** 2
            - (((x - x0) / sdx)) * ((y - y0) / sdy) * c
        )
        # f /= f.sum()
        f *= random()
        z += f
    return z


def calc_mu_g(p):
    # Average reservoir pressure
    mu_g = 3e-10 * p**2 + 1e-6 * p + 0.0133
    return mu_g


def calc_rs(p_bub, p):
    # p=average reservoir pressure
    cuda = 0
    device1 = paddle.device(f"cuda:{cuda}" if paddle.device.cuda.device_count() >= 1 else "cpu")
    rs_factor = paddle.where(
        p < p_bub,
        paddle.to_tensor(1.0).to(device1, paddle.float32),
        paddle.to_tensor(0.0).to(device1, paddle.float32),
    )
    rs = (
        (178.11**2)
        / 5.615
        * (paddle.pow(p / p_bub, 1.3) * rs_factor + (1 - rs_factor))
    )
    return rs


def calc_dp(p_bub, p_atm, p):
    dp = paddle.where(p < p_bub, p_atm - p, p_atm - p_bub)
    return dp


def calc_bg(p_bub, p_atm, p):
    # P is average reservoir pressure
    b_g = paddle.divide(1, paddle.exp(1.7e-3 * calc_dp(p_bub, p_atm, p)))
    return b_g


def calc_bo(p_bub, p_atm, CFO, p):
    # p is average reservoir pressure
    exp_term1 = paddle.where(p < p_bub, -8e-5 * (p_atm - p), -8e-5 * (p_atm - p_bub))
    exp_term2 = -CFO * paddle.where(p < p_bub, paddle.zeros_like(p), p - p_bub)
    b_o = paddle.divide(1, paddle.exp(exp_term1) * paddle.exp(exp_term2))
    return b_o


def ProgressBar(Total, Progress, BarLength=20, ProgressIcon="#", BarIcon="-"):
    try:
        # You can't have a progress bar with zero or negative length.
        if BarLength < 1:
            BarLength = 20
        # Use status variable for going to the next line after progress completion.
        Status = ""
        # Calcuting progress between 0 and 1 for percentage.
        Progress = float(Progress) / float(Total)
        # Doing this conditions at final progressing.
        if Progress >= 1.0:
            Progress = 1
            Status = "\r\n"  # Going to the next line
        # Calculating how many places should be filled
        Block = int(round(BarLength * Progress))
        # Show this
        Bar = "[{}] {:.0f}% {}".format(
            ProgressIcon * Block + BarIcon * (BarLength - Block),
            round(Progress * 100, 0),
            Status,
        )
        return Bar
    except:
        return "ERROR"


def ShowBar(Bar):
    sys.stdout.write(Bar)
    sys.stdout.flush()


def simulator_to_python(a):
    kk = a.shape[2]
    anew = []
    for i in range(kk):
        afirst = a[:, :, i]
        afirst = afirst.T
        afirst = cp.reshape(afirst, (-1, 1), "F")
        anew.append(afirst)
    return cp.vstack(anew)


def python_to_simulator(a, ny, nx, nz):
    a = cp.reshape(a, (-1, 1), "F")
    a = cp.reshape(a, (ny, nx, nz), "F")
    anew = []
    for i in range(nz):
        afirst = a[:, :, i]
        afirst = afirst.T
        anew.append(afirst)
    return cp.vstack(anew)


def rescale_linear(array, new_min, new_max):
    """Rescale an arrary linearly."""
    minimum, maximum = np.min(array), np.max(array)
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b


def rescale_linear_numpy_pytorch(array, new_min, new_max, minimum, maximum):
    """Rescale an arrary linearly."""
    m = (new_max - new_min) / (maximum - minimum)
    b = new_min - m * minimum
    return m * array + b


def rescale_linear_pytorch_numpy(array, new_min, new_max, minimum, maximum):
    """Rescale an arrary linearly."""
    m = (maximum - minimum) / (new_max - new_min)
    b = minimum - m * new_min
    return m * array + b


def read_yaml(fname):
    """Read Yaml file into a dict of parameters"""
    print(f"Read simulation plan from {fname}...")
    with open(fname, "r") as stream:
        try:
            data = yaml.safe_load(stream)
            # print(data)
        except yaml.YAMLError as exc:
            print(exc)
        return data


def Plot_2D(
    XX,
    YY,
    plt,
    nx,
    ny,
    nz,
    Truee,
    N_injw,
    N_pr,
    N_injg,
    varii,
    injectors,
    producers,
    gass,
):

    Pressz = np.reshape(Truee, (nx, ny, nz), "F")
    maxii = max(Pressz.ravel())
    minii = min(Pressz.ravel())

    avg_2d = np.mean(Pressz, axis=2)

    avg_2d[avg_2d == 0] = np.nan  # Convert zeros to NaNs
    # XX, YY = np.meshgrid(np.arange(nx),np.arange(ny))
    # plt.subplot(224)

    plt.pcolormesh(XX.T, YY.T, avg_2d, cmap="jet")
    cbar = plt.colorbar()

    if varii == "perm":
        cbar.set_label("Log K(mD)", fontsize=11)
        plt.title("Permeability Field with well locations", fontsize=11, weight="bold")
    elif varii == "water Modulus":
        cbar.set_label("water saturation", fontsize=11)
        plt.title("water saturation -Modulus", fontsize=11, weight="bold")
    elif varii == "water FLOW":
        cbar.set_label("water saturation", fontsize=11)
        plt.title("water saturation - FLOW", fontsize=11, weight="bold")
    elif varii == "water diff":
        cbar.set_label("unit", fontsize=11)
        plt.title("water saturation - (FLOW -Modulus)", fontsize=11, weight="bold")

    elif varii == "oil Modulus":
        cbar.set_label("Oil saturation", fontsize=11)
        plt.title("Oil saturation -Modulus", fontsize=11, weight="bold")

    elif varii == "oil FLOW":
        cbar.set_label("Oil saturation", fontsize=11)
        plt.title("Oil saturation - Flow", fontsize=11, weight="bold")

    elif varii == "oil diff":
        cbar.set_label("unit", fontsize=11)
        plt.title("oil saturation - (FLOW -Modulus)", fontsize=11, weight="bold")

    elif varii == "gas Modulus":
        cbar.set_label("Gas saturation", fontsize=11)
        plt.title("Gas saturation -Modulus", fontsize=11, weight="bold")

    elif varii == "gas FLOW":
        cbar.set_label("Gas saturation", fontsize=11)
        plt.title("Gas saturation -FLOW", fontsize=11, weight="bold")

    elif varii == "gas diff":
        cbar.set_label("unit", fontsize=11)
        plt.title("gas saturation - (FLOW -Modulus)", fontsize=11, weight="bold")

    elif varii == "pressure Modulus":
        cbar.set_label("pressure", fontsize=11)
        plt.title("Pressure -Modulus", fontsize=11, weight="bold")

    elif varii == "pressure FLOW":
        cbar.set_label("pressure", fontsize=11)
        plt.title("Pressure -FLOW", fontsize=11, weight="bold")

    elif varii == "pressure diff":
        cbar.set_label("unit", fontsize=11)
        plt.title("Pressure - (FLOW -Modulus)", fontsize=11, weight="bold")

    elif varii == "porosity":
        cbar.set_label("porosity", fontsize=11)
        plt.title("Porosity Field", fontsize=11, weight="bold")
    cbar.mappable.set_clim(minii, maxii)

    plt.ylabel("Y", fontsize=11)
    plt.xlabel("X", fontsize=11)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    Add_marker2(plt, XX, YY, injectors, producers, gass)


def Plot3DNorne(
    nx, ny, nz, Truee, N_injw, N_pr, N_injg, cgrid, varii, injectors, producers, gass
):

    # matplotlib.use('Agg')
    Pressz = np.reshape(Truee, (nx, ny, nz), "F")

    avg_2d = np.mean(Pressz, axis=2)

    avg_2d[avg_2d == 0] = np.nan  # Convert zeros to NaNs

    maxii = max(Pressz.ravel())
    minii = min(Pressz.ravel())
    Pressz = Pressz / maxii

    masked_Pressz = np.ma.masked_invalid(Pressz)
    colors = plt.cm.jet(masked_Pressz)
    colors[np.isnan(Pressz), :3] = 1  # set color to white for NaN values
    # alpha = np.where(np.isnan(Pressz), 0.0, 0.8) # set alpha to 0 for NaN values
    norm = mpl.colors.Normalize(vmin=minii, vmax=maxii)

    arr_3d = Pressz
    fig = plt.figure(figsize=(20, 20), dpi=200)
    ax = fig.add_subplot(221, projection="3d")

    # Shift the coordinates to center the points at the voxel locations
    x, y, z = np.indices((arr_3d.shape))
    x = x + 0.5
    y = y + 0.5
    z = z + 0.5

    # Set the colors of each voxel using a jet colormap
    # colors = plt.cm.jet(arr_3d)
    # norm = matplotlib.colors.Normalize(vmin=minii, vmax=maxii)

    # Plot each voxel and save the mappable object
    ax.voxels(arr_3d, facecolors=colors, alpha=0.5, edgecolor="none", shade=True)
    m = cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
    m.set_array([])

    # Add a colorbar for the mappable object
    # plt.colorbar(mappable)
    # Set the axis labels and title
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    # ax.set_title(titti,fontsize= 14)

    # Set axis limits to reflect the extent of each axis of the matrix
    ax.set_xlim(0, arr_3d.shape[0])
    ax.set_ylim(0, arr_3d.shape[1])
    ax.set_zlim(0, arr_3d.shape[2])
    # ax.set_zlim(0, 60)

    # Remove the grid
    ax.grid(False)

    # Set lighting to bright
    # ax.set_facecolor('white')
    # Set the aspect ratio of the plot

    ax.set_box_aspect([nx, ny, nz])

    # Set the projection type to orthogonal
    ax.set_proj_type("ortho")

    # Remove the tick labels on each axis
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # Remove the tick lines on each axis
    ax.xaxis._axinfo["tick"]["inward_factor"] = 0
    ax.xaxis._axinfo["tick"]["outward_factor"] = 0.4
    ax.yaxis._axinfo["tick"]["inward_factor"] = 0
    ax.yaxis._axinfo["tick"]["outward_factor"] = 0.4
    ax.zaxis._axinfo["tick"]["inward_factor"] = 0
    ax.zaxis._axinfo["tick"]["outward_factor"] = 0.4
    # Set the azimuth and elevation to make the plot brighter
    ax.view_init(elev=30, azim=45)

    n_inj = N_injw  # Number of injectors
    n_prod = N_pr  # Number of producers

    for mm in range(n_inj):
        usethis = injectors[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[8])
        # Define the direction of the line
        line_dir = (0, 0, (nz * 2) + 2)
        # Define the coordinates of the line end
        x_line_end = xloc + line_dir[0]
        y_line_end = yloc + line_dir[1]
        z_line_end = 0 + line_dir[2]
        ax.plot([xloc, xloc], [yloc, yloc], [0, (nz * 2) + 2], "blue", linewidth=1)
        ax.text(
            x_line_end,
            y_line_end,
            z_line_end,
            discrip,
            color="blue",
            weight="bold",
            fontsize=5,
        )

    for mm in range(N_injg):
        usethis = gass[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[8])
        # Define the direction of the line
        line_dir = (0, 0, (nz * 2) + 2)
        # Define the coordinates of the line end
        x_line_end = xloc + line_dir[0]
        y_line_end = yloc + line_dir[1]
        z_line_end = 0 + line_dir[2]
        ax.plot([xloc, xloc], [yloc, yloc], [0, (nz * 2) + 2], "red", linewidth=1)
        ax.text(
            x_line_end,
            y_line_end,
            z_line_end,
            discrip,
            color="red",
            weight="bold",
            fontsize=5,
        )

    for mm in range(n_prod):
        usethis = producers[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[8])
        # Define the direction of the line
        line_dir = (0, 0, (nz * 2) + 2)
        # Define the coordinates of the line end
        x_line_end = xloc + line_dir[0]
        y_line_end = yloc + line_dir[1]
        z_line_end = 0 + line_dir[2]
        ax.plot([xloc, xloc], [yloc, yloc], [0, (nz * 2) + 2], "g", linewidth=1)
        ax.text(
            x_line_end,
            y_line_end,
            z_line_end,
            discrip,
            color="g",
            weight="bold",
            fontsize=5,
        )

    blue_line = mlines.Line2D([], [], color="blue", linewidth=2, label="water injector")
    red_line = mlines.Line2D([], [], color="red", linewidth=2, label="Gas injector")
    green_line = mlines.Line2D(
        [], [], color="green", linewidth=2, label="oil/water/gas Producer"
    )

    # Add the legend to the plot
    ax.legend(handles=[blue_line, red_line, green_line], loc="lower left", fontsize=9)

    # Add a horizontal colorbar to the plot
    cbar = plt.colorbar(m, orientation="horizontal", shrink=0.5)
    if varii == "perm":
        cbar.set_label("Log K(mD)", fontsize=9)
        ax.set_title(
            "Permeability Field with well locations", fontsize=16, weight="bold"
        )
    elif varii == "water":
        cbar.set_label("water saturation", fontsize=9)
        ax.set_title(
            "water saturation Field with well locations", fontsize=16, weight="bold"
        )
    elif varii == "oil":
        cbar.set_label("Oil saturation", fontsize=9)
        ax.set_title(
            "Oil saturation Field with well locations", fontsize=16, weight="bold"
        )
    elif varii == "porosity":
        cbar.set_label("porosity", fontsize=9)
        ax.set_title("Porosity Field with well locations", fontsize=16, weight="bold")
    cbar.mappable.set_clim(minii, maxii)

    kxy2b = cgrid
    kxy2b[:, 6] = Truee.ravel()

    m = nx
    n = ny
    nn = nz

    Xcor = np.full((m, n, nn), np.nan)
    Ycor = np.full((m, n, nn), np.nan)
    Zcor = np.full((m, n, nn), np.nan)
    poroj = np.full((m, n, nn), np.nan)

    for j in range(kxy2b.shape[0]):
        index = int(kxy2b[j, 0] - 1)
        indey = int(kxy2b[j, 1] - 1)
        indez = int(kxy2b[j, 2] - 1)
        Xcor[index, indey, indez] = kxy2b[j, 3]
        Ycor[index, indey, indez] = kxy2b[j, 4]
        Zcor[index, indey, indez] = kxy2b[j, 5]
        poroj[index, indey, indez] = kxy2b[j, 6]

    # fig2 = plt.figure(figsize=(20, 20), dpi=100)
    # Create first subplot
    ax1 = fig.add_subplot(222, projection="3d")
    ax1.set_xlim(0, nx)
    ax1.set_ylim(0, ny)
    ax1.set_zlim(0, nz)
    ax1.set_facecolor("white")
    maxii = np.nanmax(poroj)
    minii = np.nanmin(poroj)
    colors = plt.cm.jet(np.linspace(0, 1, 256))
    colors[0, :] = (1, 1, 1, 1)  # set color for NaN values to white
    # cmap = mpl.colors.ListedColormap(colors)
    for j in range(nz):
        Xcor2D = Xcor[:, :, j]
        Ycor2D = Ycor[:, :, j]
        Zcor2D = Zcor[:, :, j]
        poroj2D = poroj[:, :, j]

        Pressz = poroj2D / maxii
        Pressz[Pressz == 0] = np.nan
        masked_Pressz = np.ma.masked_invalid(Pressz)
        colors = plt.cm.jet(masked_Pressz)
        colors[np.isnan(Pressz), :3] = 1  # set color to white for NaN values
        # alpha = np.where(np.isnan(Pressz), 0.0, 0.8) # set alpha to 0 for NaN values
        norm = mpl.colors.Normalize(vmin=minii, vmax=maxii)

        h1 = ax1.plot_surface(
            Xcor2D,
            Ycor2D,
            Zcor2D,
            cmap="jet",
            facecolors=colors,
            edgecolor="none",
            shade=True,
        )
        ax1.patch.set_facecolor("white")  # set the facecolor of the figure to white
        ax1.set_facecolor("white")

        # Add a line to the plot

    cbar = fig.colorbar(h1, orientation="horizontal", shrink=0.5)
    if varii == "perm":
        cbar.ax.set_ylabel("Log K(mD)", fontsize=9)
        ax1.set_title("Permeability Field - side view", weight="bold", fontsize=16)
    if varii == "porosity":
        cbar.ax.set_ylabel("porosity", fontsize=9)
        ax1.set_title("porosity Field - side view", weight="bold", fontsize=16)
    if varii == "oil":
        cbar.ax.set_ylabel("Oil saturation", fontsize=9)
        ax1.set_title("Oil saturation Field - side view", weight="bold", fontsize=16)
    if varii == "water":
        cbar.ax.set_ylabel("Water saturation", fontsize=9)
        ax1.set_title("Water saturation - side view", weight="bold", fontsize=16)
    cbar.mappable.set_clim(minii, maxii)

    # fig2.suptitle(r'3D Permeability NORNE FIELD [$46 \times 112 \times 22$]', fontsize=15)
    # fig2.suptitle(r'3D Permeability NORNE FIELD [$N_x = 46$, $N_y = 112$, $N_z = 22$]', fontsize=15)

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_xlim(Xcor.min(), Xcor.max())
    ax1.set_ylim(Ycor.min(), Ycor.max())
    ax1.set_zlim(Zcor.min(), Zcor.max())
    ax1.view_init(30, 30)

    n_inj = N_injw  # Number of injectors
    n_prod = N_pr  # Number of producers

    for mm in range(n_inj):
        usethis = injectors[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[8])
        # Define the direction of the line
        line_dir = (0, 0, (nz * 2) + 2)
        # Define the coordinates of the line end
        x_line_end = xloc + line_dir[0]
        y_line_end = yloc + line_dir[1]
        z_line_end = 0 + line_dir[2]
        ax1.plot([xloc, xloc], [yloc, yloc], [0, (nz * 2) + 2], "black", linewidth=2)
        ax1.text(
            x_line_end,
            y_line_end,
            z_line_end,
            discrip,
            color="black",
            weight="bold",
            fontsize=16,
        )

    for mm in range(n_prod):
        usethis = producers[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[8])
        # Define the direction of the line
        line_dir = (0, 0, (nz * 2) + 2)
        # Define the coordinates of the line end
        x_line_end = xloc + line_dir[0]
        y_line_end = yloc + line_dir[1]
        z_line_end = 0 + line_dir[2]
        ax1.plot([xloc, xloc], [yloc, yloc], [0, (nz * 2) + 2], "r", linewidth=2)
        ax1.text(
            x_line_end,
            y_line_end,
            z_line_end,
            discrip,
            color="r",
            weight="bold",
            fontsize=16,
        )

    ax1.grid(False)
    # Remove the tick labels on each axis
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_zticklabels([])

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    # Remove the tick lines on each axis
    ax1.xaxis._axinfo["tick"]["inward_factor"] = 0
    ax1.xaxis._axinfo["tick"]["outward_factor"] = 0.4
    ax1.yaxis._axinfo["tick"]["inward_factor"] = 0
    ax1.yaxis._axinfo["tick"]["outward_factor"] = 0.4
    ax1.zaxis._axinfo["tick"]["inward_factor"] = 0

    ax1 = fig.add_subplot(223, projection="3d")
    ax1.set_xlim(0, nx)
    ax1.set_ylim(0, ny)
    ax1.set_zlim(0, nz)
    ax1.set_facecolor("white")
    maxii = np.nanmax(poroj)
    minii = np.nanmin(poroj)
    colors = plt.cm.jet(np.linspace(0, 1, 256))
    colors[0, :] = (1, 1, 1, 1)  # set color for NaN values to white
    # cmap = mpl.colors.ListedColormap(colors)
    for j in range(nz):
        Xcor2D = Xcor[:, :, j]
        Ycor2D = Ycor[:, :, j]
        Zcor2D = Zcor[:, :, j]
        poroj2D = poroj[:, :, j]

        Pressz = poroj2D / maxii
        Pressz[Pressz == 0] = np.nan
        masked_Pressz = np.ma.masked_invalid(Pressz)
        colors = plt.cm.jet(masked_Pressz)
        colors[np.isnan(Pressz), :3] = 1  # set color to white for NaN values
        # alpha = np.where(np.isnan(Pressz), 0.0, 0.8) # set alpha to 0 for NaN values
        norm = mpl.colors.Normalize(vmin=minii, vmax=maxii)

        h1 = ax1.plot_surface(
            Xcor2D,
            Ycor2D,
            Zcor2D,
            cmap="jet",
            facecolors=colors,
            edgecolor="none",
            shade=True,
        )
        ax1.patch.set_facecolor("white")  # set the facecolor of the figure to white
        ax1.set_facecolor("white")

        # Add a line to the plot

    cbar = fig.colorbar(h1, orientation="horizontal", shrink=0.5)
    if varii == "perm":
        cbar.ax.set_ylabel("Log K(mD)", fontsize=9)
        ax1.set_title("Permeability Field - Top view", weight="bold", fontsize=16)
    if varii == "porosity":
        cbar.ax.set_ylabel("porosity", fontsize=9)
        ax1.set_title("porosity Field - Top view", weight="bold", fontsize=16)
    if varii == "oil":
        cbar.ax.set_ylabel("Oil saturation", fontsize=9)
        ax1.set_title("Oil saturation Field - Top view", weight="bold", fontsize=16)
    if varii == "water":
        cbar.ax.set_ylabel("Water saturation", fontsize=9)
        ax1.set_title("Water saturation - Top view", weight="bold", fontsize=16)
    cbar.mappable.set_clim(minii, maxii)

    # fig2.suptitle(r'3D Permeability NORNE FIELD [$46 \times 112 \times 22$]', fontsize=15)
    # fig2.suptitle(r'3D Permeability NORNE FIELD [$N_x = 46$, $N_y = 112$, $N_z = 22$]', fontsize=15)

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_xlim(Xcor.min(), Xcor.max())
    ax1.set_ylim(Ycor.min(), Ycor.max())
    ax1.set_zlim(Zcor.min(), Zcor.max())
    ax1.view_init(90, -90)

    n_inj = N_injw  # Number of injectors
    n_prod = N_pr  # Number of producers

    for mm in range(n_inj):
        usethis = injectors[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[8])
        # Define the direction of the line
        line_dir = (0, 0, (nz * 2) + 2)
        # Define the coordinates of the line end
        x_line_end = xloc + line_dir[0]
        y_line_end = yloc + line_dir[1]
        z_line_end = 0 + line_dir[2]
        ax1.plot([xloc, xloc], [yloc, yloc], [0, (nz * 2) + 2], "black", linewidth=2)
        ax1.text(
            x_line_end,
            y_line_end,
            z_line_end,
            discrip,
            color="black",
            weight="bold",
            fontsize=16,
        )

    for mm in range(n_prod):
        usethis = producers[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[8])
        # Define the direction of the line
        line_dir = (0, 0, (nz * 2) + 2)
        # Define the coordinates of the line end
        x_line_end = xloc + line_dir[0]
        y_line_end = yloc + line_dir[1]
        z_line_end = 0 + line_dir[2]
        ax1.plot([xloc, xloc], [yloc, yloc], [0, (nz * 2) + 2], "r", linewidth=2)
        ax1.text(
            x_line_end,
            y_line_end,
            z_line_end,
            discrip,
            color="r",
            weight="bold",
            fontsize=16,
        )

    ax1.grid(False)

    # Remove the tick labels on each axis
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_zticklabels([])

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    # Remove the tick lines on each axis
    ax1.xaxis._axinfo["tick"]["inward_factor"] = 0
    ax1.xaxis._axinfo["tick"]["outward_factor"] = 0.4
    ax1.yaxis._axinfo["tick"]["inward_factor"] = 0
    ax1.yaxis._axinfo["tick"]["outward_factor"] = 0.4
    ax1.zaxis._axinfo["tick"]["inward_factor"] = 0

    XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))
    plt.subplot(224)

    plt.pcolormesh(XX.T, YY.T, avg_2d, cmap="jet")
    cbar = plt.colorbar()

    if varii == "perm":
        cbar.ax.set_ylabel("Log K(mD)", fontsize=9)
        plt.title(r"Average Permeability Field ", fontsize=16, weight="bold")
    if varii == "porosity":
        cbar.ax.set_ylabel("porosity", fontsize=9)
        plt.title("Average porosity Field", weight="bold", fontsize=16)
    if varii == "oil":
        cbar.ax.set_ylabel("Oil saturation", fontsize=9)
        plt.title("Average Oil saturation Field - Top view", weight="bold", fontsize=16)
    if varii == "water":
        cbar.ax.set_ylabel("Water saturation", fontsize=9)
        plt.title("Average Water saturation - Top view", weight="bold", fontsize=16)

    plt.ylabel("Y", fontsize=16)
    plt.xlabel("X", fontsize=16)
    plt.axis([0, (nx - 1), 0, (ny - 1)])
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    Add_marker2(plt, XX, YY, injectors, producers, gass)

    if varii == "perm":
        fig.suptitle(
            r"3D Permeability NORNE FIELD [$N_x = 46$, $N_y = 112$, $N_z = 22$]",
            weight="bold",
            fontsize=15,
        )
    elif varii == "water":
        fig.suptitle(
            r"3D Water Saturation NORNE FIELD [$N_x = 46$, $N_y = 112$, $N_z = 22$]",
            weight="bold",
            fontsize=15,
        )
    elif varii == "oil":
        fig.suptitle(
            r"3D Oil Saturation NORNE FIELD [$N_x = 46$, $N_y = 112$, $N_z = 22$]",
            weight="bold",
            fontsize=15,
        )
    elif varii == "porosity":
        fig.suptitle(
            r"3D Porosity NORNE FIELD [$N_x = 46$, $N_y = 112$, $N_z = 22$]",
            weight="bold",
            fontsize=15,
        )

    #
    plt.savefig("All1.png")
    plt.clf()


def Plot_all_layesr(nx, ny, nz, see, injectors, producers, gass, varii):

    see[see == 0] = np.nan  # Convert zeros to NaNs
    plt.figure(figsize=(20, 20), dpi=300)
    Pressz = np.reshape(see, (nx, ny, nz), "F")
    XX, YY = np.meshgrid(np.arange(nx), np.arange(ny))

    for i in range(nz):
        plt.subplot(5, 5, i + 1)

        plt.pcolormesh(XX.T, YY.T, Pressz[:, :, i], cmap="jet")
        cbar = plt.colorbar()

        if varii == "perm":
            cbar.ax.set_ylabel("Log K(mD)", fontsize=9)
            plt.title(
                "Permeability Field Layer_" + str(i + 1), fontsize=11, weight="bold"
            )
        if varii == "porosity":
            cbar.ax.set_ylabel("porosity", fontsize=9)
            plt.title("porosity Field Layer_" + str(i + 1), weight="bold", fontsize=11)
        if varii == "oil":
            cbar.ax.set_ylabel("Oil saturation", fontsize=9)
            plt.title(
                "Oil saturation Field Layer_" + str(i + 1), weight="bold", fontsize=11
            )
        if varii == "water":
            cbar.ax.set_ylabel("Water saturation", fontsize=9)
            plt.title(
                "Water saturation Field Layer_" + str(i + 1), weight="bold", fontsize=11
            )

        plt.ylabel("Y", fontsize=11)
        plt.xlabel("X", fontsize=11)
        plt.axis([0, (nx - 1), 0, (ny - 1)])
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        Add_marker3(plt, XX, YY, injectors, producers, gass)

    if varii == "perm":
        plt.suptitle(
            r"Permeability NORNE FIELD [$N_x = 46$, $N_y = 112$, $N_z = 22$]",
            weight="bold",
            fontsize=15,
        )
    elif varii == "water":
        plt.suptitle(
            r"Water Saturation NORNE FIELD [$N_x = 46$, $N_y = 112$, $N_z = 22$]",
            weight="bold",
            fontsize=15,
        )
    elif varii == "oil":
        plt.suptitle(
            r"Oil Saturation NORNE FIELD [$N_x = 46$, $N_y = 112$, $N_z = 22$]",
            weight="bold",
            fontsize=15,
        )
    elif varii == "porosity":
        plt.suptitle(
            r"Porosity NORNE FIELD [$N_x = 46$, $N_y = 112$, $N_z = 22$]",
            weight="bold",
            fontsize=15,
        )

    #
    plt.savefig("All.png")
    plt.clf()


def Add_marker2(plt, XX, YY, injectors, producers, gass):
    """
    Function to add marker to given coordinates on a matplotlib plot

    less
    Copy code
    Parameters:
        plt: a matplotlib.pyplot object to add the markers to
        XX: a numpy array of X coordinates
        YY: a numpy array of Y coordinates
        locc: a numpy array of locations where markers need to be added

    Return:
        None
    """

    n_inj = len(injectors)  # Number of injectors
    n_prod = len(producers)  # Number of producers
    n_injg = len(gass)  # Number of gas injectors

    for mm in range(n_inj):
        usethis = injectors[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[8])

        plt.scatter(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            s=200,
            marker="v",
            color="white",
        )
        plt.text(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            discrip,
            color="black",
            weight="bold",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
        )

    for mm in range(n_injg):
        usethis = injectors[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[8])

        plt.scatter(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            s=200,
            marker="v",
            color="white",
        )
        plt.text(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            discrip,
            color="black",
            weight="bold",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
        )

    for mm in range(n_prod):
        usethis = producers[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[8])
        plt.scatter(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            s=200,
            marker="^",
            color="white",
        )
        plt.text(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            discrip,
            color="black",
            weight="bold",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
        )


def Add_marker3(plt, XX, YY, injectors, producers, gass):
    """
    Function to add marker to given coordinates on a matplotlib plot

    less
    Copy code
    Parameters:
        plt: a matplotlib.pyplot object to add the markers to
        XX: a numpy array of X coordinates
        YY: a numpy array of Y coordinates
        locc: a numpy array of locations where markers need to be added

    Return:
        None
    """

    n_inj = len(injectors)  # Number of injectors
    n_prod = len(producers)  # Number of producers
    n_injg = len(gass)  # Number of gas injectors

    for mm in range(n_inj):
        usethis = injectors[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[8])

        plt.scatter(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            s=100,
            marker="v",
            color="white",
        )
        plt.text(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            discrip,
            color="black",
            weight="bold",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=9,
        )

    for mm in range(n_injg):
        usethis = injectors[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[8])

        plt.scatter(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            s=100,
            marker="v",
            color="white",
        )
        plt.text(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            discrip,
            color="black",
            weight="bold",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=9,
        )

    for mm in range(n_prod):
        usethis = producers[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[8])
        plt.scatter(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            s=100,
            marker="^",
            color="white",
        )
        plt.text(
            XX.T[xloc - 1, yloc - 1] + 0.5,
            YY.T[xloc - 1, yloc - 1] + 0.5,
            discrip,
            color="black",
            weight="bold",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=9,
        )


def Get_Time(nx, ny, nz, steppi, steppi_indices, N):

    Timee = []
    for k in range(N):
        check = np.ones((nx, ny, nz), dtype=np.float16)
        first_column = []

        with open("FULLNORNE.RSM", "r") as f:
            # Skip the first 9 lines
            for _ in range(9):
                next(f)

            # Process the remaining lines
            for line in f:
                stripped_line = line.strip()  # Remove leading and trailing spaces
                words = stripped_line.split()  # Split line into words
                if len(words) > 0 and words[0].replace(".", "", 1).isdigit():
                    # If the first word is a number (integer or float)
                    first_column.append(float(words[0]))
                else:
                    # If the first word is not a number, it might be a header line
                    break

        # Convert list to numpy array
        np_array2 = np.array(first_column)[:-1]
        np_array2 = np_array2[steppi_indices - 1]
        unie = []
        for zz in range(len(np_array2)):
            aa = np_array2[zz] * check
            unie.append(aa)
        Time = np.stack(unie, axis=0)
        Timee.append(Time)
    Timee = np.stack(Timee, axis=0)
    return Timee


def Get_source_sink(N, nx, ny, nz, waterz, gasz, steppi, steppi_indices):
    Qw = np.zeros((1, steppi, nx, ny, nz), dtype=np.float32)
    Qg = np.zeros((1, steppi, nx, ny, nz), dtype=np.float32)
    Qo = np.zeros((1, steppi, nx, ny, nz), dtype=np.float32)

    waterz = waterz[steppi_indices - 1]
    gasz = gasz[steppi_indices - 1]

    QW = np.zeros((steppi, nx, ny, nz), dtype=np.float32)
    QG = np.zeros((steppi, nx, ny, nz), dtype=np.float32)
    QO = np.zeros((steppi, nx, ny, nz), dtype=np.float32)

    for k in range(len(steppi_indices)):
        QW[k, 25, 43, :] = waterz[k, 0]
        QW[k, 22, 14, :] = waterz[k, 1]
        QW[k, 8, 12, :] = waterz[k, 2]
        QW[k, 10, 34, :] = waterz[k, 3]
        QW[k, 28, 50, :] = waterz[k, 4]
        QW[k, 11, 84, :] = waterz[k, 5]
        QW[k, 17, 82, :] = waterz[k, 6]
        QW[k, 5, 56, :] = waterz[k, 7]
        QW[k, 35, 67, :] = waterz[k, 8]

        QG[k, 25, 43, :] = gasz[k, 0]
        QG[k, 8, 12, :] = gasz[k, 1]
        QG[k, 10, 34, :] = gasz[k, 2]
        QG[k, 28, 50, :] = gasz[k, 3]

        QO[k, 14, 30, :] = -1
        QO[k, 9, 31, :] = -1
        QO[k, 13, 33, :] = -1
        QO[k, 8, 36, :] = -1
        QO[k, 8, 45, :] = -1
        QO[k, 9, 28, :] = -1
        QO[k, 9, 23, :] = -1
        QO[k, 21, 21, :] = -1
        QO[k, 13, 27, :] = -1
        QO[k, 18, 37, :] = -1
        QO[k, 18, 53, :] = -1
        QO[k, 15, 65, :] = -1
        QO[k, 24, 36, :] = -1
        QO[k, 18, 53, :] = -1
        QO[k, 11, 71, :] = -1
        QO[k, 17, 67, :] = -1
        QO[k, 12, 66, :] = -1
        QO[k, 37, 97, :] = -1
        QO[k, 6, 63, :] = -1
        QO[k, 14, 75, :] = -1
        QO[k, 12, 66, :] = -1
        QO[k, 10, 27, :] = -1

        # Q = QW + QO + QG

    Qw[0, :, :, :, :] = QW
    Qg[0, :, :, :, :] = QG
    Qo[0, :, :, :, :] = QO
    return Qw, Qg, Qo


def Get_falt(nx, ny, nz, floatz, N):

    Fault = np.ones((nx, ny, nz), dtype=np.float16)
    flt = []
    for k in range(N):
        floatts = floatz[:, k]
        Fault[8:14, 61:81, 0:] = floatts[0]  # E_01
        Fault[7, 56:61, 0:] = floatts[1]  # E_01_F3
        Fault[9:16, 53:69, 0:15] = floatts[2]  # DE_1
        Fault[9:16, 53:69, 15:22] = floatts[3]  # DE_1LTo
        Fault[8:10, 48:53, 0:] = floatts[4]  # DE_B3
        Fault[5:8, 46:93, 0:] = floatts[5]  # DE_0
        Fault[15, 69:100, 0:] = floatts[6]  # DE_2
        Fault[5:46, 7:10, 0:11] = floatts[7]  # BC
        Fault[10:16, 44:52, 0:11] = floatts[8]  # CD
        Fault[10:16, 44:52, 11:22] = floatts[9]  # CD_To
        Fault[6:11, 38:44, 0:] = floatts[10]  # CD_B3
        Fault[5:7, 38, 0:] = floatts[11]  # CD_0
        Fault[15:19, 52:54, 0:] = floatts[12]  # CD_1
        Fault[26, 13:49, 0:] = floatts[13]  # C_01
        Fault[26, 42:48, 0:] = floatts[14]  # C_01_Ti
        Fault[24, 45:52, 10:19] = floatts[15]  # C_08
        Fault[24, 45:52, 0:10] = floatts[16]  # C_08_Ile
        Fault[24, 40:45, 0:19] = floatts[17]  # C_08_S
        Fault[24, 45:52, 19:22] = floatts[18]  # C_08_Ti
        Fault[24, 40:45, 19:22] = floatts[19]  # C_08_S_Ti
        Fault[21:25, 48:68, 0:] = floatts[20]  # C_09
        Fault[5:7, 18:22, 0:] = floatts[21]  # C_02
        Fault[23:24, 25:35, 0:] = floatts[22]  # C_04
        Fault[12:14, 41:44, 0:] = floatts[23]  # C_05
        Fault[16:19, 47:52, 0:] = floatts[24]  # C_06
        Fault[6:18, 20:22, 0:] = floatts[25]  # C_10
        Fault[25, 51:57, 0:] = floatts[26]  # C_12
        Fault[21:24, 35:39, 0:15] = floatts[27]  # C_20
        Fault[21:24, 35:39, 15:22] = floatts[28]  # C_20_LTo
        Fault[11:13, 12:24, 0:17] = floatts[29]  # C_21
        Fault[11:13, 12:24, 17:22] = floatts[30]  # C_21_Ti
        Fault[12:14, 14:18, 0:] = floatts[31]  # C_22
        Fault[19:22, 19, 0:] = floatts[32]  # C_23
        Fault[16:18, 25:27, 0:] = floatts[33]  # C_24
        Fault[20:22, 24:30, 0:] = floatts[34]  # C_25
        Fault[7:9, 27:36, 0:] = floatts[35]  # C_26
        Fault[6:8, 36:44, 0:] = floatts[36]  # C_26N
        Fault[9, 29:38, 0:] = floatts[37]  # C_27
        Fault[14:16, 38:40, 0:] = floatts[38]  # C_28
        Fault[14:16, 43:46, 0:] = floatts[39]  # C_29
        Fault[18, 47:88, 0:] = floatts[40]  # DI
        Fault[18, 44:47, 0:] = floatts[41]  # DI_S
        Fault[16:18, 55:65, 0:] = floatts[43]  # D_05
        Fault[6:10, 78:92, 0:] = floatts[43]  # EF
        Fault[28:38, 53:103, 0:] = floatts[44]  # GH
        Fault[30:33, 71:86, 0:] = floatts[45]  # G_01
        Fault[34:36, 72:75, 0:] = floatts[46]  # G_02
        Fault[33:37, 70, 0:] = floatts[47]  # G_03
        Fault[30:34, 61:65, 0:] = floatts[48]  # G_05
        Fault[30:37, 79:92, 0:] = floatts[49]  # G_07
        Fault[34, 104:107, 0:] = floatts[50]  # G_08
        Fault[30:37, 66:70, 0:] = floatts[51]  # G_09
        Fault[35:40, 64:69, 0:] = floatts[52]  # G_13
        flt.append(Fault)

    flt = np.stack(flt, axis=0)[:, None, :, :, :]
    return np.stack(flt, axis=0)


def fit_clement(tensor, target_min, target_max, tensor_min, tensor_max):
    # Rescale between target min and target max
    rescaled_tensor = tensor / tensor_max
    return rescaled_tensor


def ensemble_pytorch(
    param_perm,
    param_poro,
    param_fault,
    nx,
    ny,
    nz,
    Ne,
    effective,
    oldfolder,
    target_min,
    target_max,
    minK,
    maxK,
    minT,
    maxT,
    minP,
    maxP,
    minQ,
    maxQ,
    minQw,
    maxQw,
    minQg,
    maxQg,
    steppi,
    device,
    steppi_indices,
):

    ini_ensemble1 = np.zeros((Ne, 1, nz, nx, ny), dtype=np.float32)  # Permeability
    ini_ensemble2 = np.zeros((Ne, 1, nz, nx, ny), dtype=np.float32)  # Porosity
    ini_ensemble4 = np.zeros((Ne, 1, nz, nx, ny), dtype=np.float32)  # Fault
    ini_ensemble9 = np.zeros((Ne, 1, nz, nx, ny), dtype=np.float32)
    ini_ensemble10 = np.zeros((Ne, 1, nz, nx, ny), dtype=np.float32)

    ini_ensemble9 = 600 * np.ones((Ne, 1, nz, nx, ny))
    ini_ensemble10 = 0.2 * np.ones((Ne, 1, nz, nx, ny))
    faultz = Get_falt(nx, ny, nz, param_fault, Ne)

    for kk in range(Ne):
        a = np.reshape(param_perm[:, kk], (nx, ny, nz), "F")
        a1 = np.reshape(param_poro[:, kk], (nx, ny, nz), "F")

        for my in range(nz):
            ini_ensemble1[kk, 0, my, :, :] = a[:, :, my]  # Permeability
            ini_ensemble2[kk, 0, my, :, :] = a1[:, :, my]  # Porosity
            ini_ensemble4[kk, 0, my, :, :] = faultz[kk, 0, :, :, my]  # fault

    # Initial_pressure
    ini_ensemble9 = fit_clement(ini_ensemble9, target_min, target_max, minP, maxP)

    # Permeability
    ini_ensemble1 = fit_clement(ini_ensemble1, target_min, target_max, minK, maxK)

    # ini_ensemble = paddle.from_numpy(ini_ensemble).to(device, dtype=paddle.float32)
    inn = {
        "perm": paddle.from_numpy(ini_ensemble1).to(device, paddle.float32),
        "fault": paddle.from_numpy(ini_ensemble4).to(device, dtype=paddle.float32),
        "Phi": paddle.from_numpy(ini_ensemble2).to(device, dtype=paddle.float32),
        "Pini": paddle.from_numpy(ini_ensemble9).to(device, dtype=paddle.float32),
        "Swini": paddle.from_numpy(ini_ensemble10).to(device, dtype=paddle.float32),
    }
    return inn


def Geta_all(
    folder,
    nx,
    ny,
    nz,
    effective,
    oldfolder,
    check,
    string_Jesus,
    steppi,
    steppi_indices,
):
    os.chdir(folder)

    # os.system(string_Jesus)
    check = np.ones((nx, ny, nz), dtype=np.float32)
    first_column = []

    with open("FULLNORNE2.RSM", "r") as f:
        # Skip the first 9 lines
        for _ in range(9):
            next(f)

        # Process the remaining lines
        for line in f:
            stripped_line = line.strip()  # Remove leading and trailing spaces
            words = stripped_line.split()  # Split line into words
            if len(words) > 0 and words[0].replace(".", "", 1).isdigit():
                # If the first word is a number (integer or float)
                first_column.append(float(words[0]))
            else:
                # If the first word is not a number, it might be a header line
                break

    # Convert list to numpy array
    np_array2 = np.array(first_column)[:-1]
    np_array2 = np_array2[steppi_indices - 1].ravel()
    unie = []
    for zz in range(steppi):
        aa = np_array2[zz] * check
        unie.append(aa)
    Time = np.stack(unie, axis=0)

    indices = np.where(effective == 1)[0]
    size1 = len(indices)

    pressure = []
    swat = []
    sgas = []

    for ii in range(1, 247):
        filename2 = f"FULLNORNE.A{str(ii).zfill(4)}"  # zfill pads the string with zeroes to reach a length of 4
        os.remove(filename2)

    steppi_indices = steppi_indices.flatten()
    for i in steppi_indices:  # 248 because range's end is exclusive
        filename = f"FULLNORNE.F{str(i).zfill(4)}"  # zfill pads the string with zeroes to reach a length of 4

        with open(filename, "r") as f:
            pressure_flag = False
            all_values = []
            for line in f:
                stripped_line = line.strip()  # Remove leading and trailing spaces
                if "PRESSURE" in stripped_line and "REAL" in stripped_line:
                    pressure_flag = True  # Start capturing pressure data
                elif pressure_flag:  # We're in pressure data capture mode
                    if stripped_line[
                        0
                    ].isdigit():  # If the stripped line starts with a digit
                        values = list(map(float, stripped_line.split()))
                        all_values.extend(values)  # Add values to the list
                    else:
                        break  # Stop capturing pressure data

        # Convert list to numpy array
        np_array = np.array(all_values)
        vector1 = np.zeros((nx * ny * nz, 1))
        vector1[indices] = rzz(
            np_array.reshape(-1, 1), (size1,), order=1, preserve_range=True
        )
        vector1 = np.reshape(vector1, (nx, ny, nz), "F")
        pressure.append(vector1)

        with open(filename, "r") as f:
            pressure_flag = False
            all_values = []
            for line in f:
                stripped_line = line.strip()  # Remove leading and trailing spaces
                if "SWAT" in stripped_line and "REAL" in stripped_line:
                    pressure_flag = True  # Start capturing pressure data
                elif pressure_flag:  # We're in pressure data capture mode
                    if stripped_line[
                        0
                    ].isdigit():  # If the stripped line starts with a digit
                        values = list(map(float, stripped_line.split()))
                        all_values.extend(values)  # Add values to the list
                    else:
                        break  # Stop capturing swat data

        # Convert list to numpy array
        np_array = np.array(all_values)
        vector1 = np.zeros((nx * ny * nz, 1))
        vector1[indices] = rzz(
            np_array.reshape(-1, 1), (size1,), order=1, preserve_range=True
        )
        vector1 = np.reshape(vector1, (nx, ny, nz), "F")
        swat.append(vector1)

        with open(filename, "r") as f:
            pressure_flag = False
            all_values = []
            for line in f:
                stripped_line = line.strip()  # Remove leading and trailing spaces
                if "SGAS" in stripped_line and "REAL" in stripped_line:
                    pressure_flag = True  # Start capturing pressure data
                elif pressure_flag:  # We're in pressure data capture mode
                    if stripped_line[
                        0
                    ].isdigit():  # If the stripped line starts with a digit
                        values = list(map(float, stripped_line.split()))
                        all_values.extend(values)  # Add values to the list
                    else:
                        break  # Stop capturing sgas data
        try:
            os.remove(filename)
        except OSError as e:
            print(f"Error: {filename} : {e.strerror}")
        # Convert list to numpy array
        np_array = np.array(all_values)
        vector1 = np.zeros((nx * ny * nz, 1))
        vector1[indices] = rzz(
            np_array.reshape(-1, 1), (size1,), order=1, preserve_range=True
        )
        vector1 = np.reshape(vector1, (nx, ny, nz), "F")
        sgas.append(vector1)

    sgas = np.stack(sgas, axis=0)
    pressure = np.stack(pressure, axis=0)
    swat = np.stack(swat, axis=0)

    """
    This section is for the FTM vaues
    """
    float_parameters = []
    file_path = "multflt.dat"
    with open(file_path, "r") as file:
        for line in file:
            # Use split to separate the string into parts.
            split_line = line.split()

            # Ensure there are at least two elements in the split line
            if len(split_line) >= 2:
                try:
                    # The second element (index 1) should be the float you're interested in.
                    float_parameter = float(split_line[1])
                    float_parameters.append(float_parameter)
                except ValueError:
                    # Handle cases where the second element can't be converted to a float.
                    pass
            else:
                pass
    floatts = np.hstack(float_parameters)

    Fault = np.ones((nx, ny, nz), dtype=np.float16)
    Fault[8:14, 61:81, 0:] = floatts[0]  # E_01
    Fault[7, 56:61, 0:] = floatts[1]  # E_01_F3
    Fault[9:16, 53:69, 0:15] = floatts[2]  # DE_1
    Fault[9:16, 53:69, 15:22] = floatts[3]  # DE_1LTo
    Fault[8:10, 48:53, 0:] = floatts[4]  # DE_B3
    Fault[5:8, 46:93, 0:] = floatts[5]  # DE_0
    Fault[15, 69:100, 0:] = floatts[6]  # DE_2
    Fault[5:46, 7:10, 0:11] = floatts[7]  # BC
    Fault[10:16, 44:52, 0:11] = floatts[8]  # CD
    Fault[10:16, 44:52, 11:22] = floatts[9]  # CD_To
    Fault[6:11, 38:44, 0:] = floatts[10]  # CD_B3
    Fault[5:7, 38, 0:] = floatts[11]  # CD_0
    Fault[15:19, 52:54, 0:] = floatts[12]  # CD_1
    Fault[26, 13:49, 0:] = floatts[13]  # C_01
    Fault[26, 42:48, 0:] = floatts[14]  # C_01_Ti
    Fault[24, 45:52, 10:19] = floatts[15]  # C_08
    Fault[24, 45:52, 0:10] = floatts[16]  # C_08_Ile
    Fault[24, 40:45, 0:19] = floatts[17]  # C_08_S
    Fault[24, 45:52, 19:22] = floatts[18]  # C_08_Ti
    Fault[24, 40:45, 19:22] = floatts[19]  # C_08_S_Ti
    Fault[21:25, 48:68, 0:] = floatts[20]  # C_09
    Fault[5:7, 18:22, 0:] = floatts[21]  # C_02
    Fault[23:24, 25:35, 0:] = floatts[22]  # C_04
    Fault[12:14, 41:44, 0:] = floatts[23]  # C_05
    Fault[16:19, 47:52, 0:] = floatts[24]  # C_06
    Fault[6:18, 20:22, 0:] = floatts[25]  # C_10
    Fault[25, 51:57, 0:] = floatts[26]  # C_12
    Fault[21:24, 35:39, 0:15] = floatts[27]  # C_20
    Fault[21:24, 35:39, 15:22] = floatts[28]  # C_20_LTo
    Fault[11:13, 12:24, 0:17] = floatts[29]  # C_21
    Fault[11:13, 12:24, 17:22] = floatts[30]  # C_21_Ti
    Fault[12:14, 14:18, 0:] = floatts[31]  # C_22
    Fault[19:22, 19, 0:] = floatts[32]  # C_23
    Fault[16:18, 25:27, 0:] = floatts[33]  # C_24
    Fault[20:22, 24:30, 0:] = floatts[34]  # C_25
    Fault[7:9, 27:36, 0:] = floatts[35]  # C_26
    Fault[6:8, 36:44, 0:] = floatts[36]  # C_26N
    Fault[9, 29:38, 0:] = floatts[37]  # C_27
    Fault[14:16, 38:40, 0:] = floatts[38]  # C_28
    Fault[14:16, 43:46, 0:] = floatts[39]  # C_29
    Fault[18, 47:88, 0:] = floatts[40]  # DI
    Fault[18, 44:47, 0:] = floatts[41]  # DI_S
    Fault[16:18, 55:65, 0:] = floatts[43]  # D_05
    Fault[6:10, 78:92, 0:] = floatts[43]  # EF
    Fault[28:38, 53:103, 0:] = floatts[44]  # GH
    Fault[30:33, 71:86, 0:] = floatts[45]  # G_01
    Fault[34:36, 72:75, 0:] = floatts[46]  # G_02
    Fault[33:37, 70, 0:] = floatts[47]  # G_03
    Fault[30:34, 61:65, 0:] = floatts[48]  # G_05
    Fault[30:37, 79:92, 0:] = floatts[49]  # G_07
    Fault[34, 104:107, 0:] = floatts[50]  # G_08
    Fault[30:37, 66:70, 0:] = floatts[51]  # G_09
    Fault[35:40, 64:69, 0:] = floatts[52]  # G_13
    os.chdir(oldfolder)
    return pressure, swat, sgas, Time, Fault


def Get_RSM_FLOW(oldfolder, N, steppi, steppi_indices):
    WOIL1 = np.zeros((steppi, N))
    WOIL2 = np.zeros((steppi, N))
    WOIL3 = np.zeros((steppi, N))
    WOIL4 = np.zeros((steppi, N))
    WOIL5 = np.zeros((steppi, N))
    WOIL6 = np.zeros((steppi, N))
    WOIL7 = np.zeros((steppi, N))
    WOIL8 = np.zeros((steppi, N))
    WOIL9 = np.zeros((steppi, N))
    WOIL10 = np.zeros((steppi, N))
    WOIL11 = np.zeros((steppi, N))
    WOIL12 = np.zeros((steppi, N))
    WOIL13 = np.zeros((steppi, N))
    WOIL14 = np.zeros((steppi, N))
    WOIL15 = np.zeros((steppi, N))
    WOIL16 = np.zeros((steppi, N))
    WOIL17 = np.zeros((steppi, N))
    WOIL18 = np.zeros((steppi, N))
    WOIL19 = np.zeros((steppi, N))
    WOIL20 = np.zeros((steppi, N))
    WOIL21 = np.zeros((steppi, N))
    WOIL22 = np.zeros((steppi, N))

    WWATER1 = np.zeros((steppi, N))
    WWATER2 = np.zeros((steppi, N))
    WWATER3 = np.zeros((steppi, N))
    WWATER4 = np.zeros((steppi, N))
    WWATER5 = np.zeros((steppi, N))
    WWATER6 = np.zeros((steppi, N))
    WWATER7 = np.zeros((steppi, N))
    WWATER8 = np.zeros((steppi, N))
    WWATER9 = np.zeros((steppi, N))
    WWATER10 = np.zeros((steppi, N))
    WWATER11 = np.zeros((steppi, N))
    WWATER12 = np.zeros((steppi, N))
    WWATER13 = np.zeros((steppi, N))
    WWATER14 = np.zeros((steppi, N))
    WWATER15 = np.zeros((steppi, N))
    WWATER16 = np.zeros((steppi, N))
    WWATER17 = np.zeros((steppi, N))
    WWATER18 = np.zeros((steppi, N))
    WWATER19 = np.zeros((steppi, N))
    WWATER20 = np.zeros((steppi, N))
    WWATER21 = np.zeros((steppi, N))
    WWATER22 = np.zeros((steppi, N))

    WGAS1 = np.zeros((steppi, N))
    WGAS2 = np.zeros((steppi, N))
    WGAS3 = np.zeros((steppi, N))
    WGAS4 = np.zeros((steppi, N))
    WGAS5 = np.zeros((steppi, N))
    WGAS6 = np.zeros((steppi, N))
    WGAS7 = np.zeros((steppi, N))
    WGAS8 = np.zeros((steppi, N))
    WGAS9 = np.zeros((steppi, N))
    WGAS10 = np.zeros((steppi, N))
    WGAS11 = np.zeros((steppi, N))
    WGAS12 = np.zeros((steppi, N))
    WGAS13 = np.zeros((steppi, N))
    WGAS14 = np.zeros((steppi, N))
    WGAS15 = np.zeros((steppi, N))
    WGAS16 = np.zeros((steppi, N))
    WGAS17 = np.zeros((steppi, N))
    WGAS18 = np.zeros((steppi, N))
    WGAS19 = np.zeros((steppi, N))
    WGAS20 = np.zeros((steppi, N))
    WGAS21 = np.zeros((steppi, N))
    WGAS22 = np.zeros((steppi, N))

    WWINJ1 = np.zeros((steppi, N))
    WWINJ2 = np.zeros((steppi, N))
    WWINJ3 = np.zeros((steppi, N))
    WWINJ4 = np.zeros((steppi, N))
    WWINJ5 = np.zeros((steppi, N))
    WWINJ6 = np.zeros((steppi, N))
    WWINJ7 = np.zeros((steppi, N))
    WWINJ8 = np.zeros((steppi, N))
    WWINJ9 = np.zeros((steppi, N))

    WGASJ1 = np.zeros((steppi, N))
    WGASJ2 = np.zeros((steppi, N))
    WGASJ3 = np.zeros((steppi, N))
    WGASJ4 = np.zeros((steppi, N))

    steppii = 246
    for i in range(N):
        f = "Realisation_"
        folder = f + str(i)
        os.chdir(folder)
        ## IMPORT FOR OIL
        A2oilsim = pd.read_csv("FULLNORNE2.RSM", skiprows=1545, sep="\s+", header=None)

        B_1BHoilsim = A2oilsim[5].values[:steppii]
        B_1Hoilsim = A2oilsim[6].values[:steppii]
        B_2Hoilsim = A2oilsim[7].values[:steppii]
        B_3Hoilsim = A2oilsim[8].values[:steppii]
        B_4BHoilsim = A2oilsim[9].values[:steppii]

        A22oilsim = pd.read_csv("FULLNORNE2.RSM", skiprows=1801, sep="\s+", header=None)
        B_4DHoilsim = A22oilsim[1].values[:steppii]
        B_4Hoilsim = A22oilsim[2].values[:steppii]
        D_1CHoilsim = A22oilsim[3].values[:steppii]
        D_1Hoilsim = A22oilsim[4].values[:steppii]
        D_2Hoilsim = A22oilsim[5].values[:steppii]
        D_3AHoilsim = A22oilsim[6].values[:steppii]
        D_3BHoilsim = A22oilsim[7].values[:steppii]
        D_4AHoilsim = A22oilsim[8].values[:steppii]
        D_4Hoilsim = A22oilsim[9].values[:steppii]

        A222oilsim = pd.read_csv(
            "FULLNORNE2.RSM", skiprows=2057, sep="\s+", header=None
        )

        E_1Hoilsim = A222oilsim[1].values[:steppii]
        E_2AHoilsim = A222oilsim[2].values[:steppii]
        E_2Hoilsim = A222oilsim[3].values[:steppii]
        E_3AHoilsim = A222oilsim[4].values[:steppii]
        E_3CHoilsim = A222oilsim[5].values[:steppii]
        E_3Hoilsim = A222oilsim[6].values[:steppii]
        E_4AHoilsim = A222oilsim[7].values[:steppii]
        K_3Hoilsim = A222oilsim[8].values[:steppii]

        WOIL1[:, i] = B_1BHoilsim.ravel()[steppi_indices - 1]
        WOIL2[:, i] = B_1Hoilsim.ravel()[steppi_indices - 1]
        WOIL3[:, i] = B_2Hoilsim.ravel()[steppi_indices - 1]
        WOIL4[:, i] = B_3Hoilsim.ravel()[steppi_indices - 1]
        WOIL5[:, i] = B_4BHoilsim.ravel()[steppi_indices - 1]
        WOIL6[:, i] = B_4DHoilsim.ravel()[steppi_indices - 1]
        WOIL7[:, i] = B_4Hoilsim.ravel()[steppi_indices - 1]
        WOIL8[:, i] = D_1CHoilsim.ravel()[steppi_indices - 1]
        WOIL9[:, i] = D_1Hoilsim.ravel()[steppi_indices - 1]
        WOIL10[:, i] = D_2Hoilsim.ravel()[steppi_indices - 1]
        WOIL11[:, i] = D_3AHoilsim.ravel()[steppi_indices - 1]
        WOIL12[:, i] = D_3BHoilsim.ravel()[steppi_indices - 1]
        WOIL13[:, i] = D_4AHoilsim.ravel()[steppi_indices - 1]
        WOIL14[:, i] = D_4Hoilsim.ravel()[steppi_indices - 1]
        WOIL15[:, i] = E_1Hoilsim.ravel()[steppi_indices - 1]
        WOIL16[:, i] = E_2AHoilsim.ravel()[steppi_indices - 1]
        WOIL17[:, i] = E_2Hoilsim.ravel()[steppi_indices - 1]
        WOIL18[:, i] = E_3AHoilsim.ravel()[steppi_indices - 1]
        WOIL19[:, i] = E_3CHoilsim.ravel()[steppi_indices - 1]
        WOIL20[:, i] = E_3Hoilsim.ravel()[steppi_indices - 1]
        WOIL21[:, i] = E_4AHoilsim.ravel()[steppi_indices - 1]
        WOIL22[:, i] = K_3Hoilsim.ravel()[steppi_indices - 1]

        ##IMPORT FOR WATER
        A2watersim = pd.read_csv(
            "FULLNORNE2.RSM", skiprows=2313, sep="\s+", header=None
        )
        B_1BHwatersim = A2watersim[9].values[:steppi]

        A22watersim = pd.read_csv(
            "FULLNORNE2.RSM", skiprows=2569, sep="\s+", header=None
        )
        B_1Hwatersim = A22watersim[1].values[:steppii]
        B_2Hwatersim = A22watersim[2].values[:steppii]
        B_3Hwatersim = A22watersim[3].values[:steppii]
        B_4BHwatersim = A22watersim[4].values[:steppii]
        B_4DHwatersim = A22watersim[5].values[:steppii]
        B_4Hwatersim = A22watersim[6].values[:steppii]
        D_1CHwatersim = A22watersim[7].values[:steppii]
        D_1Hwatersim = A22watersim[8].values[:steppii]
        D_2Hwatersim = A22watersim[9].values[:steppii]

        A222watersim = pd.read_csv(
            "FULLNORNE2.RSM", skiprows=2825, sep="\s+", header=None
        )
        D_3AHwatersim = A222watersim[1].values[:steppii]
        D_3BHwatersim = A222watersim[2].values[:steppii]
        D_4AHwatersim = A222watersim[3].values[:steppii]
        D_4Hwatersim = A222watersim[4].values[:steppii]
        E_1Hwatersim = A222watersim[5].values[:steppii]
        E_2AHwatersim = A222watersim[6].values[:steppii]
        E_2Hwatersim = A222watersim[7].values[:steppii]
        E_3AHwatersim = A222watersim[8].values[:steppii]
        E_3CHwatersim = A222watersim[9].values[:steppii]

        A222watersim = pd.read_csv(
            "FULLNORNE2.RSM", skiprows=3081, sep="\s+", header=None
        )
        E_3Hwatersim = A222watersim[1].values[:steppii]
        E_4AHwatersim = A222watersim[2].values[:steppii]
        K_3Hwatersim = A222watersim[3].values[:steppii]

        WWATER1[:, i] = B_1BHwatersim.ravel()[steppi_indices - 1]
        WWATER2[:, i] = B_1Hwatersim.ravel()[steppi_indices - 1]
        WWATER3[:, i] = B_2Hwatersim.ravel()[steppi_indices - 1]
        WWATER4[:, i] = B_3Hwatersim.ravel()[steppi_indices - 1]
        WWATER5[:, i] = B_4BHwatersim.ravel()[steppi_indices - 1]
        WWATER6[:, i] = B_4DHwatersim.ravel()[steppi_indices - 1]
        WWATER7[:, i] = B_4Hwatersim.ravel()[steppi_indices - 1]
        WWATER8[:, i] = D_1CHwatersim.ravel()[steppi_indices - 1]
        WWATER9[:, i] = D_1Hwatersim.ravel()[steppi_indices - 1]
        WWATER10[:, i] = D_2Hwatersim.ravel()[steppi_indices - 1]
        WWATER11[:, i] = D_3AHwatersim.ravel()[steppi_indices - 1]
        WWATER12[:, i] = D_3BHwatersim.ravel()[steppi_indices - 1]
        WWATER13[:, i] = D_4AHwatersim.ravel()[steppi_indices - 1]
        WWATER14[:, i] = D_4Hwatersim.ravel()[steppi_indices - 1]
        WWATER15[:, i] = E_1Hwatersim.ravel()[steppi_indices - 1]
        WWATER16[:, i] = E_2AHwatersim.ravel()[steppi_indices - 1]
        WWATER17[:, i] = E_2Hwatersim.ravel()[steppi_indices - 1]
        WWATER18[:, i] = E_3AHwatersim.ravel()[steppi_indices - 1]
        WWATER19[:, i] = E_3CHwatersim.ravel()[steppi_indices - 1]

        # if len(E_3Hwatersim<steppi):
        #     E_3Hwatersim = np.append(E_3Hwatersim.ravel(), 0)
        # if len(E_4AHwatersim<steppi):
        #     E_4AHwatersim = np.append(E_4AHwatersim.ravel(), 0)
        # if len(K_3Hwatersim<steppi):
        #     K_3Hwatersim = np.append(K_3Hwatersim.ravel(), 0)
        WWATER20[:, i] = E_3Hwatersim.ravel()[steppi_indices - 1]
        WWATER21[:, i] = E_4AHwatersim.ravel()[steppi_indices - 1]
        WWATER22[:, i] = K_3Hwatersim.ravel()[steppi_indices - 1]

        ## GAS PRODUCTION RATE
        A2gassim = pd.read_csv("FULLNORNE2.RSM", skiprows=1033, sep="\s+", header=None)
        B_1BHgassim = A2gassim[1].values[:steppii]
        B_1Hgassim = A2gassim[2].values[:steppii]
        B_2Hgassim = A2gassim[3].values[:steppii]
        B_3Hgassim = A2gassim[4].values[:steppii]
        B_4BHgassim = A2gassim[5].values[:steppii]
        B_4DHgassim = A2gassim[6].values[:steppii]
        B_4Hgassim = A2gassim[7].values[:steppii]
        D_1CHgassim = A2gassim[8].values[:steppii]
        D_1Hgassim = A2gassim[9].values[:steppii]

        A22gassim = pd.read_csv("FULLNORNE2.RSM", skiprows=1289, sep="\s+", header=None)
        D_2Hgassim = A22gassim[1].values[:steppii]
        D_3AHgassim = A22gassim[2].values[:steppii]
        D_3BHgassim = A22gassim[3].values[:steppii]
        D_4AHgassim = A22gassim[4].values[:steppii]
        D_4Hgassim = A22gassim[5].values[:steppii]
        E_1Hgassim = A22gassim[6].values[:steppii]
        E_2AHgassim = A22gassim[7].values[:steppii]
        E_2Hgassim = A22gassim[8].values[:steppii]
        E_3AHgassim = A22gassim[9].values[:steppii]

        A222gassim = pd.read_csv(
            "FULLNORNE2.RSM", skiprows=1545, sep="\s+", header=None
        )
        E_3CHgassim = A222gassim[1].values[:steppii]
        E_3Hgassim = A222gassim[2].values[:steppii]
        E_4AHgassim = A222gassim[3].values[:steppii]
        K_3Hgassim = A222gassim[4].values[:steppii]

        WGAS1[:, i] = B_1BHgassim.ravel()[steppi_indices - 1]
        WGAS2[:, i] = B_1Hgassim.ravel()[steppi_indices - 1]
        WGAS3[:, i] = B_2Hgassim.ravel()[steppi_indices - 1]
        WGAS4[:, i] = B_3Hgassim.ravel()[steppi_indices - 1]
        WGAS5[:, i] = B_4BHgassim.ravel()[steppi_indices - 1]
        WGAS6[:, i] = B_4DHgassim.ravel()[steppi_indices - 1]
        WGAS7[:, i] = B_4Hgassim.ravel()[steppi_indices - 1]
        WGAS8[:, i] = D_1CHgassim.ravel()[steppi_indices - 1]
        WGAS9[:, i] = D_1Hgassim.ravel()[steppi_indices - 1]
        WGAS10[:, i] = D_2Hgassim.ravel()[steppi_indices - 1]
        WGAS11[:, i] = D_3AHgassim.ravel()[steppi_indices - 1]
        WGAS12[:, i] = D_3BHgassim.ravel()[steppi_indices - 1]
        WGAS13[:, i] = D_4AHgassim.ravel()[steppi_indices - 1]
        WGAS14[:, i] = D_4Hgassim.ravel()[steppi_indices - 1]
        WGAS15[:, i] = E_1Hgassim.ravel()[steppi_indices - 1]
        WGAS16[:, i] = E_2AHgassim.ravel()[steppi_indices - 1]
        WGAS17[:, i] = E_2Hgassim.ravel()[steppi_indices - 1]
        WGAS18[:, i] = E_3AHgassim.ravel()[steppi_indices - 1]
        WGAS19[:, i] = E_3CHgassim.ravel()[steppi_indices - 1]
        WGAS20[:, i] = E_3Hgassim.ravel()[steppi_indices - 1]
        WGAS21[:, i] = E_4AHgassim.ravel()[steppi_indices - 1]
        WGAS22[:, i] = K_3Hgassim.ravel()[steppi_indices - 1]

        ## WATER INJECTOR RATE
        A2wir = pd.read_csv("FULLNORNE2.RSM", skiprows=2057, sep="\s+", header=None)
        C_1Hwaterinjsim = A2wir[9].values[:steppi]

        A22wir = pd.read_csv("FULLNORNE2.RSM", skiprows=2313, sep="\s+", header=None)
        C_2Hwaterinjsim = A22wir[1].values[:steppii]
        C_3Hwaterinjsim = A22wir[2].values[:steppii]
        C_4AHwaterinjsim = A22wir[3].values[:steppii]
        C_4Hwaterinjsim = A22wir[4].values[:steppii]
        F_1Hwaterinjsim = A22wir[5].values[:steppii]
        F_2Hwaterinjsim = A22wir[6].values[:steppii]
        F_3Hwaterinjsim = A22wir[7].values[:steppii]
        F_4Hwaterinjsim = A22wir[8].values[:steppii]

        WWINJ1[:, i] = C_1Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ2[:, i] = C_2Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ3[:, i] = C_3Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ4[:, i] = C_4AHwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ5[:, i] = C_4Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ6[:, i] = F_1Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ7[:, i] = F_2Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ8[:, i] = F_3Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ9[:, i] = F_4Hwaterinjsim.ravel()[steppi_indices - 1]

        ## GAS INJECTOR RATE
        A2gir = pd.read_csv("FULLNORNE2.RSM", skiprows=777, sep="\s+", header=None)
        C_1Hgasinjsim = A2gir[6].values[:steppii]
        C_3Hgasinjsim = A2gir[7].values[:steppii]
        C_4AHgasinjsim = A2gir[8].values[:steppii]
        C_4Hgasinjsim = A2gir[9].values[:steppii]

        WGASJ1[:, i] = C_1Hgasinjsim.ravel()[steppi_indices - 1]
        WGASJ2[:, i] = C_3Hgasinjsim.ravel()[steppi_indices - 1]
        WGASJ3[:, i] = C_4AHgasinjsim.ravel()[steppi_indices - 1]
        WGASJ4[:, i] = C_4Hgasinjsim.ravel()[steppi_indices - 1]

        # Return to the root folder
        os.chdir(oldfolder)

    OIL = {
        "B_1BH": WOIL1,
        "B_1H": WOIL2,
        "B_2H": WOIL3,
        "B_3H": WOIL4,
        "B_4BH": WOIL5,
        "B_4DH": WOIL6,
        "B_4H": WOIL7,
        "D_1CH": WOIL8,
        "D_1H": WOIL9,
        "D_2H": WOIL10,
        "D_3AH": WOIL11,
        "D_3BH": WOIL12,
        "D_4AH": WOIL13,
        "D_4H": WOIL14,
        "E_1H": WOIL15,
        "E_2AH": WOIL16,
        "E_2H": WOIL17,
        "E_3AH": WOIL18,
        "E_3CH": WOIL19,
        "E_3H": WOIL20,
        "E_4AH": WOIL21,
        "K_3H": WOIL22,
    }

    OIL = {k: abs(v) for k, v in OIL.items()}

    WATER = {
        "B_1BH": WWATER1,
        "B_1H": WWATER2,
        "B_2H": WWATER3,
        "B_3H": WWATER4,
        "B_4BH": WWATER5,
        "B_4DH": WWATER6,
        "B_4H": WWATER7,
        "D_1CH": WWATER8,
        "D_1H": WWATER9,
        "D_2H": WWATER10,
        "D_3AH": WWATER11,
        "D_3BH": WWATER12,
        "D_4AH": WWATER13,
        "D_4H": WWATER14,
        "E_1H": WWATER15,
        "E_2AH": WWATER16,
        "E_2H": WWATER17,
        "E_3AH": WWATER18,
        "E_3CH": WWATER19,
        "E_3H": WWATER20,
        "E_4AH": WWATER21,
        "K_3H": WWATER22,
    }

    WATER = {k: abs(v) for k, v in WATER.items()}

    GAS = {
        "B_1BH": WGAS1,
        "B_1H": WGAS2,
        "B_2H": WGAS3,
        "B_3H": WGAS4,
        "B_4BH": WGAS5,
        "B_4DH": WGAS6,
        "B_4H": WGAS7,
        "D_1CH": WGAS8,
        "D_1H": WGAS9,
        "D_2H": WGAS10,
        "D_3AH": WGAS11,
        "D_3BH": WGAS12,
        "D_4AH": WGAS13,
        "D_4H": WGAS14,
        "E_1H": WGAS15,
        "E_2AH": WGAS16,
        "E_2H": WGAS17,
        "E_3AH": WGAS18,
        "E_3CH": WGAS19,
        "E_3H": WGAS20,
        "E_4AH": WGAS21,
        "K_3H": WGAS22,
    }

    GAS = {k: abs(v) for k, v in GAS.items()}

    WATERINJ = {
        "C_1H": WWINJ1,
        "C_2H": WWINJ2,
        "C_3H": WWINJ3,
        "C_4AH": WWINJ4,
        "C_4H": WWINJ5,
        "F_1H": WWINJ6,
        "F_2H": WWINJ7,
        "F_3H": WWINJ8,
        "F_4H": WWINJ9,
    }

    WATERINJ = {k: abs(v) for k, v in WATERINJ.items()}

    GASINJ = {"C_1H": WGASJ1, "C_3H": WGASJ2, "C_4AH": WGASJ3, "C_4H": WGASJ4}

    GASINJ = {k: abs(v) for k, v in GASINJ.items()}
    return OIL, WATER, GAS, WATERINJ, GASINJ


def Get_predicted_data(OIL, WATER, GAS, WATERINJ, GASINJ):
    matrices = [OIL[key] for key in sorted(OIL.keys())]
    matrix_columns = [np.hsplit(matrix, matrix.shape[1]) for matrix in matrices]
    result_columns = [np.vstack(column_pair) for column_pair in zip(*matrix_columns)]
    result_oil = np.hstack(result_columns)

    matrices = [WATER[key] for key in sorted(WATER.keys())]
    matrix_columns = [np.hsplit(matrix, matrix.shape[1]) for matrix in matrices]
    result_columns = [np.vstack(column_pair) for column_pair in zip(*matrix_columns)]
    result_water = np.hstack(result_columns)

    matrices = [GAS[key] for key in sorted(GAS.keys())]
    matrix_columns = [np.hsplit(matrix, matrix.shape[1]) for matrix in matrices]
    result_columns = [np.vstack(column_pair) for column_pair in zip(*matrix_columns)]
    result_gas = np.hstack(result_columns)

    matrices = [WATERINJ[key] for key in sorted(WATERINJ.keys())]
    matrix_columns = [np.hsplit(matrix, matrix.shape[1]) for matrix in matrices]
    result_columns = [np.vstack(column_pair) for column_pair in zip(*matrix_columns)]
    result_winj = np.hstack(result_columns)

    matrices = [GASINJ[key] for key in sorted(GASINJ.keys())]
    matrix_columns = [np.hsplit(matrix, matrix.shape[1]) for matrix in matrices]
    result_columns = [np.vstack(column_pair) for column_pair in zip(*matrix_columns)]
    result_ginj = np.hstack(result_columns)

    predicted_data = np.nan_to_num(
        np.vstack((result_oil, result_water, result_gas, result_winj, result_ginj))
    )
    return predicted_data


def copy_files(source_dir, dest_dir):
    files = os.listdir(source_dir)
    for file in files:
        shutil.copy(os.path.join(source_dir, file), dest_dir)


def save_files(perm, poro, perm2, dest_dir, oldfolder):
    os.chdir(dest_dir)
    filename1 = "permx" + ".dat"
    np.savetxt(
        filename1,
        perm,
        fmt="%.4f",
        delimiter=" \t",
        newline="\n",
        header="PERMX",
        footer="/",
        comments="",
    )

    filename2 = "porosity" + ".dat"
    np.savetxt(
        filename2,
        poro,
        fmt="%.4f",
        delimiter=" \t",
        newline="\n",
        header="PORO",
        footer="/",
        comments="",
    )

    my_array = perm2.ravel()
    my_array_index = 0

    # read the file
    with open("multflt.dat", "r") as file:
        lines = file.readlines()
    for i, line in enumerate(lines):
        if line.strip() == "MULTFLT":
            # do nothing if this is the MULTFLT line
            continue
        else:
            # replace numerical values on other lines
            parts = line.split(" ")
            if (
                len(parts) > 1
                and parts[1].replace(".", "", 1).replace("/", "").isdigit()
            ):
                parts[1] = str(my_array[my_array_index])
                lines[i] = " ".join(parts)
                my_array_index += 1

    # write the file
    with open("multflt.dat", "w") as file:
        file.writelines(lines)

    os.chdir(oldfolder)


def Run_simulator(dest_dir, oldfolder, string_jesus, string_jesus2):
    os.chdir(dest_dir)
    os.system(string_jesus)
    os.system(string_jesus2)
    os.chdir(oldfolder)


def pytorch_cupy(tensor):
    dx = to_dlpack(tensor)
    cx = cp.fromDlpack(dx)
    return cx


def cupy_pytorch(cx):
    tx2 = from_dlpack(cx.toDlpack())
    return tx2


def convert_back(rescaled_tensor, target_min, target_max, min_val, max_val):
    # If min_val and max_val are equal, just set the tensor to min_val
    # if min_val == max_val:
    #     original_tensor = np.full(rescaled_tensor.shape, min_val)
    # else:
    #     original_tensor = min_val + (rescaled_tensor - target_min) * (max_val - min_val) / (target_max - target_min)

    return rescaled_tensor * max_val


def replace_nans_and_infs(tensor, value=0.0):
    tensor[paddle.isnan(tensor) | paddle.isinf(tensor)] = value
    return tensor


def scale_clement(tensor, target_min, target_max):
    tensor[np.isnan(tensor)] = 0  # Replace NaN with 0
    tensor[np.isinf(tensor)] = 0  # Replace infinity with 0

    min_val = np.min(tensor)
    max_val = np.max(tensor)

    # # Check for tensor_max equal to tensor_min to avoid division by zero
    # if max_val == min_val:
    #     rescaled_tensor = np.full(tensor.shape, (target_max + target_min) / 2)  # Set the tensor to the average of target_min and target_max
    # else:
    rescaled_tensor = tensor / max_val

    return min_val, max_val, rescaled_tensor


# Define the threshold as the largest finite representable number for np.float32
threshold = np.finfo(np.float32).max


def replace_large_and_invalid_values(arr, placeholder=0.0):
    """
    Replaces large values, infinities, and NaNs in a numpy array with a placeholder.

    Parameters:
    - arr: numpy array to be modified.
    - placeholder: value to replace invalid entries with.

    Returns:
    - Modified numpy array.
    """
    invalid_indices = (np.isnan(arr)) | (np.isinf(arr)) | (np.abs(arr) > threshold)
    arr[invalid_indices] = placeholder
    return arr


def clean_dict_arrays(data_dict):
    """
    Replaces large numbers, infinities, and NaNs in numpy arrays in a dictionary with a placeholder.

    Parameters:
    - data_dict: dictionary containing numpy arrays.

    Returns:
    - Dictionary with cleaned numpy arrays.
    """
    for key in data_dict:
        data_dict[key] = replace_large_and_invalid_values(data_dict[key])
    return data_dict


def clip_and_convert_to_float32(array):
    """
    Clips the values in the input array to the representable range of np.float32
    and then converts the array to dtype np.float32.

    Parameters:
    - array: numpy array to be clipped and converted.

    Returns:
    - Clipped and converted numpy array.
    """
    max_float32 = np.finfo(np.float32).max
    min_float32 = np.finfo(np.float32).min

    array_clipped = np.clip(array, min_float32, max_float32)
    # array_clipped = round_array_to_4dp(array_clipped)
    return array_clipped.astype(np.float32)


def clip_and_convert_to_float3(array):
    """
    Clips the values in the input array to the representable range of np.float32
    and then converts the array to dtype np.float32.

    Parameters:
    - array: numpy array to be clipped and converted.

    Returns:
    - Clipped and converted numpy array.
    """
    max_float32 = np.finfo(np.float32).max
    min_float32 = np.finfo(np.float32).min

    array_clipped = np.clip(array, min_float32, max_float32)
    # array_clipped = round_array_to_4dp(array_clipped)
    return array_clipped.astype(np.float32)


def Make_correct(array):
    """
    This function rearranges the dimensions of a 5D numpy array.

    Given an array of shape (a, b, c, d, e), it returns an array
    of shape (a, b, d, e, c). The function swaps the third and
    fifth dimensions.

    Parameters:
    - array (numpy.ndarray): A 5D numpy array to be rearranged.

    Returns:
    - numpy.ndarray: A 5D numpy array with rearranged dimensions.
    """

    # Initialize a new 5D array with swapped axes 2 and 4
    new_array = np.zeros(
        (array.shape[0], array.shape[1], array.shape[3], array.shape[4], array.shape[2])
    )

    # Loop through the first dimension
    for kk in range(array.shape[0]):
        # Initialize a 4D array for temporary storage
        perm_big = np.zeros(
            (array.shape[1], array.shape[3], array.shape[4], array.shape[2])
        )

        # Loop through the second dimension
        for mum in range(array.shape[1]):
            # Initialize a 3D array for deeper storage
            mum1 = np.zeros((array.shape[3], array.shape[4], array.shape[2]))

            # Loop through the third (original) dimension
            for i in range(array.shape[2]):
                # Rearrange the data from the original array
                mum1[:, :, i] = array[kk, :, :, :, :][mum, :, :, :][i, :, :]

            # Update the 4D array
            perm_big[mum, :, :, :] = mum1

        # Update the new 5D array
        new_array[kk, :, :, :, :] = perm_big

    return new_array


def Forward_model_ensemble(
    N,
    x_true,
    steppi,
    min_inn_fcn,
    max_inn_fcn,
    target_min,
    target_max,
    minK,
    maxK,
    minT,
    maxT,
    minP,
    maxP,
    modelP1,
    modelW1,
    modelG1,
    device,
    modelP2,
    min_out_fcn,
    max_out_fcn,
    Time,
    effectiveuse,
    Trainmoe,
    num_cores,
    pred_type,
    oldfolder,
):

    #### ===================================================================== ####
    #                     RESERVOIR SIMULATOR WITH MODULUS
    #
    #### ===================================================================== ####
    """
    Parameters:
    -----------
    N : int
        Number of ensemble samples.

    x_true : dict
        A dictionary containing the ground truth data for perm, Phi, fault, Pini, and Swini.

    steppi : int
        Number of time steps.

    min_inn_fcn, max_inn_fcn : float
        Minimum and maximum values for input scaling.

    target_min, target_max : float
        Minimum and maximum target values for scaling.

    minK, maxK, minT, maxT, minP, maxP : float
        Min-max values for permeability, time, and pressure.

    modelP1, modelW1, modelG1, modelP2 : paddle.nn.Layer
        Trained PyTorch models for predicting pressure, water saturation, gas saturation, and the secondary pressure model respectively.

    device : paddle.device
        The device (CPU or GPU) where the PyTorch models will run.

    min_out_fcn, max_out_fcn : float
        Minimum and maximum values for output scaling.

    Time : array-like
        Reservoir simulation time values.

    Returns:
    --------
    ouut_p : array-like
        Predicted outputs based on the input ensemble using the trained models.

    Description:
    ------------
    The function starts by predicting pressure, water saturation, and gas
    saturation using the input x_true and the trained models modelP1,
    modelW1, and modelG1. Then, the function processes the predictions,
    and scales and converts them back to their original values.
    Afterward, a new set of inputs for a secondary model (modelP2) is prepared
    using the processed data. Finally, the function predicts outputs using the
    modelP2 and returns the predicted values.
    """

    pressure = []
    Swater = []
    Sgas = []

    for clem in range(N):
        temp = {
            "perm": x_true["perm"][clem, :, :, :, :][None, :, :, :, :],
            "Phi": x_true["Phi"][clem, :, :, :, :][None, :, :, :, :],
            "fault": x_true["fault"][clem, :, :, :, :][None, :, :, :, :],
            "Pini": x_true["Pini"][clem, :, :, :, :][None, :, :, :, :],
            "Swini": x_true["Swini"][clem, :, :, :, :][None, :, :, :, :],
        }

        with paddle.no_grad():
            ouut_p1 = modelP1(temp)["pressure"]
            ouut_s1 = modelW1(temp)["water_sat"]
            ouut_sg1 = modelG1(temp)["gas_sat"]

        pressure.append(ouut_p1)
        Swater.append(ouut_s1)
        Sgas.append(ouut_sg1)

        del temp
        paddle.device.cuda.empty_cache()

    pressure = paddle.vstack(pressure).detach().cpu().numpy()
    pressure = Make_correct(pressure)
    Swater = paddle.vstack(Swater).detach().cpu().numpy()
    Swater = Make_correct(Swater)
    Sgas = paddle.vstack(Sgas).detach().cpu().numpy()
    Sgas = Make_correct(Sgas)
    Soil = np.ones_like(Swater) - (Swater + Sgas)

    # The inputs are at the normal scale

    perm = convert_back(
        x_true["perm"].detach().cpu().numpy(), target_min, target_max, minK, maxK
    )
    perm = Make_correct(perm)
    # Time = convert_back(x_true["Time"].detach().cpu().numpy() ,target_min,target_max,minT,maxT)
    # Time = Make_correct(Time)
    pressure = convert_back(pressure, target_min, target_max, minP, maxP)

    if Trainmoe == 1:
        innn = np.zeros((N, 90, steppi))
    else:
        innn = np.zeros((N, steppi, 90))

    for i in range(N):

        ##INPUTS
        permuse = perm[i, 0, :, :, :]
        # Permeability
        a1 = np.mean(permuse[14, 30, :]) * np.ones((steppi, 1))
        a2 = np.mean(permuse[9, 31, :]) * np.ones((steppi, 1))
        a3 = np.mean(permuse[13, 33, :]) * np.ones((steppi, 1))
        a5 = np.mean(permuse[8, 36, :]) * np.ones((steppi, 1))
        a6 = np.mean(permuse[8, 45, :]) * np.ones((steppi, 1))
        a7 = np.mean(permuse[9, 28, :]) * np.ones((steppi, 1))
        a8 = np.mean(permuse[9, 23, :]) * np.ones((steppi, 1))
        a9 = np.mean(permuse[21, 21, :]) * np.ones((steppi, 1))
        a10 = np.mean(permuse[13, 27, :]) * np.ones((steppi, 1))
        a11 = np.mean(permuse[18, 37, :]) * np.ones((steppi, 1))
        a12 = np.mean(permuse[18, 53, :]) * np.ones((steppi, 1))
        a13 = np.mean(permuse[15, 65, :]) * np.ones((steppi, 1))
        a14 = np.mean(permuse[24, 36, :]) * np.ones((steppi, 1))
        a15 = np.mean(permuse[18, 53, :]) * np.ones((steppi, 1))
        a16 = np.mean(permuse[11, 71, :]) * np.ones((steppi, 1))
        a17 = np.mean(permuse[17, 67, :]) * np.ones((steppi, 1))
        a18 = np.mean(permuse[12, 66, :]) * np.ones((steppi, 1))
        a19 = np.mean(permuse[37, 97, :]) * np.ones((steppi, 1))
        a20 = np.mean(permuse[6, 63, :]) * np.ones((steppi, 1))
        a21 = np.mean(permuse[14, 75, :]) * np.ones((steppi, 1))
        a22 = np.mean(permuse[12, 66, :]) * np.ones((steppi, 1))
        a23 = np.mean(permuse[10, 27, :]) * np.ones((steppi, 1))

        permxx = np.hstack(
            (
                a1,
                a2,
                a3,
                a5,
                a6,
                a7,
                a8,
                a9,
                a10,
                a11,
                a12,
                a13,
                a14,
                a15,
                a16,
                a17,
                a18,
                a19,
                a20,
                a21,
                a22,
                a23,
            )
        )

        presure_use = pressure[i, :, :, :, :]
        gas_use = Sgas[i, :, :, :, :]
        water_use = Swater[i, :, :, :, :]
        Time_use = Time[i, :, :, :, :]

        a1 = np.zeros((steppi, 1))
        a2 = np.zeros((steppi, 22))
        a3 = np.zeros((steppi, 22))
        a4 = np.zeros((steppi, 1))

        for k in range(steppi):
            uniep = presure_use[k, :, :, :]
            permuse = uniep
            a1[k, 0] = np.mean(permuse * effectiveuse)

            uniegas = gas_use[k, :, :, :]
            permuse = uniegas
            a2[k, 0] = np.mean(permuse[14, 30, :])
            a2[k, 1] = np.mean(permuse[9, 31, :])
            a2[k, 2] = np.mean(permuse[13, 33, :])
            a2[k, 3] = np.mean(permuse[8, 36, :])
            a2[k, 4] = np.mean(permuse[8, 45, :])
            a2[k, 5] = np.mean(permuse[9, 28, :])
            a2[k, 6] = np.mean(permuse[9, 23, :])
            a2[k, 7] = np.mean(permuse[21, 21, :])
            a2[k, 8] = np.mean(permuse[13, 27, :])
            a2[k, 9] = np.mean(permuse[18, 37, :])
            a2[k, 10] = np.mean(permuse[18, 53, :])
            a2[k, 11] = np.mean(permuse[15, 65, :])
            a2[k, 12] = np.mean(permuse[24, 36, :])
            a2[k, 13] = np.mean(permuse[18, 54, :])
            a2[k, 14] = np.mean(permuse[11, 71, :])
            a2[k, 15] = np.mean(permuse[17, 67, :])
            a2[k, 16] = np.mean(permuse[12, 66, :])
            a2[k, 17] = np.mean(permuse[37, 97, :])
            a2[k, 18] = np.mean(permuse[6, 63, :])
            a2[k, 19] = np.mean(permuse[14, 75, :])
            a2[k, 20] = np.mean(permuse[12, 66, :])
            a2[k, 21] = np.mean(permuse[10, 27, :])

            uniewater = water_use[k, :, :, :]
            permuse = uniewater
            a3[k, 0] = np.mean(permuse[14, 30, :])
            a3[k, 1] = np.mean(permuse[9, 31, :])
            a3[k, 2] = np.mean(permuse[13, 33, :])
            a3[k, 3] = np.mean(permuse[8, 36, :])
            a3[k, 4] = np.mean(permuse[8, 45, :])
            a3[k, 5] = np.mean(permuse[9, 28, :])
            a3[k, 6] = np.mean(permuse[9, 23, :])
            a3[k, 7] = np.mean(permuse[21, 21, :])
            a3[k, 8] = np.mean(permuse[13, 27, :])
            a3[k, 9] = np.mean(permuse[18, 37, :])
            a3[k, 10] = np.mean(permuse[18, 53, :])
            a3[k, 11] = np.mean(permuse[15, 65, :])
            a3[k, 12] = np.mean(permuse[24, 36, :])
            a3[k, 13] = np.mean(permuse[18, 54, :])
            a3[k, 14] = np.mean(permuse[11, 71, :])
            a3[k, 15] = np.mean(permuse[17, 67, :])
            a3[k, 16] = np.mean(permuse[12, 66, :])
            a3[k, 17] = np.mean(permuse[37, 97, :])
            a3[k, 18] = np.mean(permuse[6, 63, :])
            a3[k, 19] = np.mean(permuse[14, 75, :])
            a3[k, 20] = np.mean(permuse[12, 66, :])
            a3[k, 21] = np.mean(permuse[10, 27, :])

            unietime = Time_use[k, :, :, :]
            permuse = unietime
            a4[k, 0] = permuse[0, 0, 0]

        inn1 = np.hstack((permxx, a1, 1 - (a2 + a3), a2, a3, a4))

        inn1 = fit_clement(inn1, target_min, target_max, min_inn_fcn, max_inn_fcn)
        if Trainmoe == 1:
            innn[i, :, :] = inn1.T
        else:
            innn[i, :, :] = inn1

    if Trainmoe == 1:
        innn = paddle.from_numpy(innn).to(device, paddle.float32)
        ouut_p = []
        # x_true = ensemblepy
        for clem in range(N):
            temp = {"X": innn[clem, :, :][None, :, :]}

            with paddle.no_grad():
                ouut_p1 = modelP2(temp)["Y"]
                ouut_p1 = convert_back(
                    ouut_p1.detach().cpu().numpy(),
                    target_min,
                    target_max,
                    min_out_fcn,
                    max_out_fcn,
                )

            ouut_p.append(ouut_p1)
            del temp
            paddle.device.cuda.empty_cache()
        ouut_p = np.vstack(ouut_p)
        ouut_p = np.transpose(ouut_p, (0, 2, 1))
    else:
        innn = np.vstack(innn)
        cluster_all = np.genfromtxt("../ML_MACHINE/clustersizescost.dat", dtype="float")
        cluster_all = np.reshape(cluster_all, (-1, 1), "F")

        # clemes = []
        # for ib in range(66):
        #     progressBar = "\rInference(CCR) - Peaceman well model: " + ProgressBar(66-1, ib-1, 66-1)
        #     ShowBar(progressBar)
        #     time.sleep(1)
        #     see = PREDICTION_CCR__MACHINE(ib,int(cluster_all[ib,:]),innn,innn.shape[1],\
        #                 "../ML_MACHINE",oldfolder,pred_type,3)
        #     clemes.append(see)
        # progressBar = "\rInference(CCR) - Peaceman well model: " + ProgressBar(66-1, ib, 66-1)
        # ShowBar(progressBar)
        # time.sleep(1)

        clemes = Parallel(n_jobs=6, backend="loky", verbose=10)(
            delayed(PREDICTION_CCR__MACHINE)(
                ib,
                int(cluster_all[ib, :]),
                innn,
                innn.shape[1],
                "../ML_MACHINE",
                oldfolder,
                pred_type,
                3,
            )
            for ib in range(66)
        )

        ouut_p = np.array(Split_Matrix(np.hstack(clemes), N))
        ouut_p = convert_back(ouut_p, target_min, target_max, min_out_fcn, max_out_fcn)

    ouut_p[ouut_p <= 0] = 0
    sim = []
    for zz in range(ouut_p.shape[0]):
        use = np.reshape(ouut_p[zz, :, :66], (-1, 1), "F")
        use[use < 0] = 0
        sim.append(use)
    sim = np.hstack(sim)
    return sim, ouut_p, pressure, Swater, Sgas, Soil


def Split_Matrix(matrix, sizee):
    x_split = np.split(matrix, sizee, axis=0)
    return x_split


def Get_data_FFNN(
    oldfolder, N, pressure, Sgas, Swater, perm, Time, steppi, steppi_indices
):

    ouut = np.zeros((N, steppi, 66))
    innn = np.zeros((N, steppi, 90))
    steppii = 246
    for i in range(N):
        WOIL1 = np.zeros((steppi, 22))
        WWATER1 = np.zeros((steppi, 22))
        WGAS1 = np.zeros((steppi, 22))
        WWINJ1 = np.zeros((steppi, 9))
        WGASJ1 = np.zeros((steppi, 4))

        # f = 'Realisation'
        # folder = f + str(i)
        folder = to_absolute_path("../RUNS/Realisation" + str(i))
        os.chdir(folder)
        ## IMPORT FOR OIL
        A2oilsim = pd.read_csv("FULLNORNE2.RSM", skiprows=1545, sep="\s+", header=None)

        B_1BHoilsim = A2oilsim[5].values[:steppii]
        B_1Hoilsim = A2oilsim[6].values[:steppii]
        B_2Hoilsim = A2oilsim[7].values[:steppii]
        B_3Hoilsim = A2oilsim[8].values[:steppii]
        B_4BHoilsim = A2oilsim[9].values[:steppii]

        A22oilsim = pd.read_csv("FULLNORNE2.RSM", skiprows=1801, sep="\s+", header=None)
        B_4DHoilsim = A22oilsim[1].values[:steppii]
        B_4Hoilsim = A22oilsim[2].values[:steppii]
        D_1CHoilsim = A22oilsim[3].values[:steppii]
        D_1Hoilsim = A22oilsim[4].values[:steppii]
        D_2Hoilsim = A22oilsim[5].values[:steppii]
        D_3AHoilsim = A22oilsim[6].values[:steppii]
        D_3BHoilsim = A22oilsim[7].values[:steppii]
        D_4AHoilsim = A22oilsim[8].values[:steppii]
        D_4Hoilsim = A22oilsim[9].values[:steppii]

        A222oilsim = pd.read_csv(
            "FULLNORNE2.RSM", skiprows=2057, sep="\s+", header=None
        )

        E_1Hoilsim = A222oilsim[1].values[:steppii]
        E_2AHoilsim = A222oilsim[2].values[:steppii]
        E_2Hoilsim = A222oilsim[3].values[:steppii]
        E_3AHoilsim = A222oilsim[4].values[:steppii]
        E_3CHoilsim = A222oilsim[5].values[:steppii]
        E_3Hoilsim = A222oilsim[6].values[:steppii]
        E_4AHoilsim = A222oilsim[7].values[:steppii]
        K_3Hoilsim = A222oilsim[8].values[:steppii]

        WOIL1[:, 0] = B_1BHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 1] = B_1Hoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 2] = B_2Hoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 3] = B_3Hoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 4] = B_4BHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 5] = B_4DHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 6] = B_4Hoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 7] = D_1CHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 8] = D_1Hoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 9] = D_2Hoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 10] = D_3AHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 11] = D_3BHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 12] = D_4AHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 13] = D_4Hoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 14] = E_1Hoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 15] = E_2AHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 16] = E_2Hoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 17] = E_3AHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 18] = E_3CHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 19] = E_3Hoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 20] = E_4AHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 21] = K_3Hoilsim.ravel()[steppi_indices - 1]

        ##IMPORT FOR WATER
        A2watersim = pd.read_csv(
            "FULLNORNE2.RSM", skiprows=2313, sep="\s+", header=None
        )
        B_1BHwatersim = A2watersim[9].values[:steppii]

        A22watersim = pd.read_csv(
            "FULLNORNE2.RSM", skiprows=2569, sep="\s+", header=None
        )
        B_1Hwatersim = A22watersim[1].values[:steppii]
        B_2Hwatersim = A22watersim[2].values[:steppii]
        B_3Hwatersim = A22watersim[3].values[:steppii]
        B_4BHwatersim = A22watersim[4].values[:steppii]
        B_4DHwatersim = A22watersim[5].values[:steppii]
        B_4Hwatersim = A22watersim[6].values[:steppii]
        D_1CHwatersim = A22watersim[7].values[:steppii]
        D_1Hwatersim = A22watersim[8].values[:steppii]
        D_2Hwatersim = A22watersim[9].values[:steppii]

        A222watersim = pd.read_csv(
            "FULLNORNE2.RSM", skiprows=2825, sep="\s+", header=None
        )
        D_3AHwatersim = A222watersim[1].values[:steppii]
        D_3BHwatersim = A222watersim[2].values[:steppii]
        D_4AHwatersim = A222watersim[3].values[:steppii]
        D_4Hwatersim = A222watersim[4].values[:steppii]
        E_1Hwatersim = A222watersim[5].values[:steppii]
        E_2AHwatersim = A222watersim[6].values[:steppii]
        E_2Hwatersim = A222watersim[7].values[:steppii]
        E_3AHwatersim = A222watersim[8].values[:steppii]
        E_3CHwatersim = A222watersim[9].values[:steppii]

        A222watersim = pd.read_csv(
            "FULLNORNE2.RSM", skiprows=3081, sep="\s+", header=None
        )
        E_3Hwatersim = A222watersim[1].values[:steppii]
        E_4AHwatersim = A222watersim[2].values[:steppii]
        K_3Hwatersim = A222watersim[3].values[:steppii]

        WWATER1[:, 0] = B_1BHwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 1] = B_1Hwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 2] = B_2Hwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 3] = B_3Hwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 4] = B_4BHwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 5] = B_4DHwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 6] = B_4Hwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 7] = D_1CHwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 8] = D_1Hwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 9] = D_2Hwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 10] = D_3AHwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 11] = D_3BHwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 12] = D_4AHwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 13] = D_4Hwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 14] = E_1Hwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 15] = E_2AHwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 16] = E_2Hwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 17] = E_3AHwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 18] = E_3CHwatersim.ravel()[steppi_indices - 1]

        WWATER1[:, 19] = E_3Hwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 20] = E_4AHwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 21] = K_3Hwatersim.ravel()[steppi_indices - 1]

        ## GAS PRODUCTION RATE
        A2gassim = pd.read_csv("FULLNORNE2.RSM", skiprows=1033, sep="\s+", header=None)
        B_1BHgassim = A2gassim[1].values[:steppii]
        B_1Hgassim = A2gassim[2].values[:steppii]
        B_2Hgassim = A2gassim[3].values[:steppii]
        B_3Hgassim = A2gassim[4].values[:steppii]
        B_4BHgassim = A2gassim[5].values[:steppii]
        B_4DHgassim = A2gassim[6].values[:steppii]
        B_4Hgassim = A2gassim[7].values[:steppii]
        D_1CHgassim = A2gassim[8].values[:steppii]
        D_1Hgassim = A2gassim[9].values[:steppii]

        A22gassim = pd.read_csv("FULLNORNE2.RSM", skiprows=1289, sep="\s+", header=None)
        D_2Hgassim = A22gassim[1].values[:steppii]
        D_3AHgassim = A22gassim[2].values[:steppii]
        D_3BHgassim = A22gassim[3].values[:steppii]
        D_4AHgassim = A22gassim[4].values[:steppii]
        D_4Hgassim = A22gassim[5].values[:steppii]
        E_1Hgassim = A22gassim[6].values[:steppii]
        E_2AHgassim = A22gassim[7].values[:steppii]
        E_2Hgassim = A22gassim[8].values[:steppii]
        E_3AHgassim = A22gassim[9].values[:steppii]

        A222gassim = pd.read_csv(
            "FULLNORNE2.RSM", skiprows=1545, sep="\s+", header=None
        )
        E_3CHgassim = A222gassim[1].values[:steppii]
        E_3Hgassim = A222gassim[2].values[:steppii]
        E_4AHgassim = A222gassim[3].values[:steppii]
        K_3Hgassim = A222gassim[4].values[:steppii]

        WGAS1[:, 0] = B_1BHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 1] = B_1Hgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 2] = B_2Hgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 3] = B_3Hgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 4] = B_4BHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 5] = B_4DHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 6] = B_4Hgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 7] = D_1CHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 8] = D_1Hgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 9] = D_2Hgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 10] = D_3AHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 11] = D_3BHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 12] = D_4AHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 13] = D_4Hgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 14] = E_1Hgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 15] = E_2AHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 16] = E_2Hgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 17] = E_3AHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 18] = E_3CHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 19] = E_3Hgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 20] = E_4AHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 21] = K_3Hgassim.ravel()[steppi_indices - 1]

        ## WATER INJECTOR RATE
        A2wir = pd.read_csv("FULLNORNE2.RSM", skiprows=2057, sep="\s+", header=None)
        C_1Hwaterinjsim = A2wir[9].values[:steppii]

        A22wir = pd.read_csv("FULLNORNE2.RSM", skiprows=2313, sep="\s+", header=None)
        C_2Hwaterinjsim = A22wir[1].values[:steppii]
        C_3Hwaterinjsim = A22wir[2].values[:steppii]
        C_4AHwaterinjsim = A22wir[3].values[:steppii]
        C_4Hwaterinjsim = A22wir[4].values[:steppii]
        F_1Hwaterinjsim = A22wir[5].values[:steppii]
        F_2Hwaterinjsim = A22wir[6].values[:steppii]
        F_3Hwaterinjsim = A22wir[7].values[:steppii]
        F_4Hwaterinjsim = A22wir[8].values[:steppii]

        WWINJ1[:, 0] = C_1Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ1[:, 1] = C_2Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ1[:, 2] = C_3Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ1[:, 3] = C_4AHwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ1[:, 4] = C_4Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ1[:, 5] = F_1Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ1[:, 6] = F_2Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ1[:, 7] = F_3Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ1[:, 8] = F_4Hwaterinjsim.ravel()[steppi_indices - 1]

        ## GAS INJECTOR RATE
        A2gir = pd.read_csv("FULLNORNE2.RSM", skiprows=777, sep="\s+", header=None)
        C_1Hgasinjsim = A2gir[6].values[:steppii]
        C_3Hgasinjsim = A2gir[7].values[:steppii]
        C_4AHgasinjsim = A2gir[8].values[:steppii]
        C_4Hgasinjsim = A2gir[9].values[:steppii]

        WGASJ1[:, 0] = C_1Hgasinjsim.ravel()[steppi_indices - 1]
        WGASJ1[:, 1] = C_3Hgasinjsim.ravel()[steppi_indices - 1]
        WGASJ1[:, 2] = C_4AHgasinjsim.ravel()[steppi_indices - 1]
        WGASJ1[:, 3] = C_4Hgasinjsim.ravel()[steppi_indices - 1]

        # Return to the root folder
        # out = np.hstack((WOIL1,WWATER1, WGAS1, WWINJ1, WGASJ1))
        out = np.hstack((WOIL1, WWATER1, WGAS1))
        out[out <= 0] = 0
        ouut[i, :, :] = out

        ##INPUTS

        permuse = perm[i, 0, :, :, :]
        a1 = np.mean(permuse[14, 30, :]) * np.ones((steppi, 1))
        a2 = np.mean(permuse[9, 31, :]) * np.ones((steppi, 1))
        a3 = np.mean(permuse[13, 33, :]) * np.ones((steppi, 1))
        a5 = np.mean(permuse[8, 36, :]) * np.ones((steppi, 1))
        a6 = np.mean(permuse[8, 45, :]) * np.ones((steppi, 1))
        a7 = np.mean(permuse[9, 28, :]) * np.ones((steppi, 1))
        a8 = np.mean(permuse[9, 23, :]) * np.ones((steppi, 1))
        a9 = np.mean(permuse[21, 21, :]) * np.ones((steppi, 1))
        a10 = np.mean(permuse[13, 27, :]) * np.ones((steppi, 1))
        a11 = np.mean(permuse[18, 37, :]) * np.ones((steppi, 1))
        a12 = np.mean(permuse[18, 53, :]) * np.ones((steppi, 1))
        a13 = np.mean(permuse[15, 65, :]) * np.ones((steppi, 1))
        a14 = np.mean(permuse[24, 36, :]) * np.ones((steppi, 1))
        a15 = np.mean(permuse[18, 53, :]) * np.ones((steppi, 1))
        a16 = np.mean(permuse[11, 71, :]) * np.ones((steppi, 1))
        a17 = np.mean(permuse[17, 67, :]) * np.ones((steppi, 1))
        a18 = np.mean(permuse[12, 66, :]) * np.ones((steppi, 1))
        a19 = np.mean(permuse[37, 97, :]) * np.ones((steppi, 1))
        a20 = np.mean(permuse[6, 63, :]) * np.ones((steppi, 1))
        a21 = np.mean(permuse[14, 75, :]) * np.ones((steppi, 1))
        a22 = np.mean(permuse[12, 66, :]) * np.ones((steppi, 1))
        a23 = np.mean(permuse[10, 27, :]) * np.ones((steppi, 1))

        permxx = np.hstack(
            (
                a1,
                a2,
                a3,
                a5,
                a6,
                a7,
                a8,
                a9,
                a10,
                a11,
                a12,
                a13,
                a14,
                a15,
                a16,
                a17,
                a18,
                a19,
                a20,
                a21,
                a22,
                a23,
            )
        )

        presure_use = pressure[i, :, :, :, :]
        gas_use = Sgas[i, :, :, :, :]
        water_use = Swater[i, :, :, :, :]
        Time_use = Time[i, :, :, :, :]

        a1 = np.zeros((steppi, 1))
        a2 = np.zeros((steppi, 22))
        a3 = np.zeros((steppi, 22))
        a4 = np.zeros((steppi, 1))

        for k in range(steppi):
            uniep = presure_use[k, :, :, :]
            permuse = uniep
            # a1[k,0] = np.mean(permuse[14, 30,:])
            # a1[k,1] = np.mean(permuse[9, 31, :])
            # a1[k,2] = np.mean(permuse[13, 33,:])
            # a1[k,3] = np.mean(permuse[8, 36, :])
            # a1[k,4] = np.mean(permuse[8, 45, :])
            # a1[k,5] = np.mean(permuse[9, 28,:])
            # a1[k,6] = np.mean(permuse[9, 23, :])
            # a1[k,7] = np.mean(permuse[21, 21, :])
            # a1[k,8] = np.mean(permuse[13, 27, :])
            # a1[k,9] = np.mean(permuse[18, 37, :])
            # a1[k,10] = np.mean(permuse[18, 53, :])
            # a1[k,11] = np.mean(permuse[15, 65, :])
            # a1[k,12] = np.mean(permuse[24, 36, :])
            # a1[k,13] = np.mean(permuse[18, 54, :])
            # a1[k,14] = np.mean(permuse[11, 71, :])
            # a1[k,15] = np.mean(permuse[17, 67, :])
            # a1[k,16] = np.mean(permuse[12, 66, :])
            # a1[k,17] = np.mean(permuse[37, 97, :])
            # a1[k,18] = np.mean(permuse[6, 63, :])
            # a1[k,19] = np.mean(permuse[14, 75, :])
            # a1[k,20] = np.mean(permuse[12, 66, :])
            # a1[k,21] = np.mean(permuse[10, 27, :])

            a1[k, 0] = np.mean(permuse)

            uniegas = gas_use[k, :, :, :]
            permuse = uniegas
            a2[k, 0] = np.mean(permuse[14, 30, :])
            a2[k, 1] = np.mean(permuse[9, 31, :])
            a2[k, 2] = np.mean(permuse[13, 33, :])
            a2[k, 3] = np.mean(permuse[8, 36, :])
            a2[k, 4] = np.mean(permuse[8, 45, :])
            a2[k, 5] = np.mean(permuse[9, 28, :])
            a2[k, 6] = np.mean(permuse[9, 23, :])
            a2[k, 7] = np.mean(permuse[21, 21, :])
            a2[k, 8] = np.mean(permuse[13, 27, :])
            a2[k, 9] = np.mean(permuse[18, 37, :])
            a2[k, 10] = np.mean(permuse[18, 53, :])
            a2[k, 11] = np.mean(permuse[15, 65, :])
            a2[k, 12] = np.mean(permuse[24, 36, :])
            a2[k, 13] = np.mean(permuse[18, 54, :])
            a2[k, 14] = np.mean(permuse[11, 71, :])
            a2[k, 15] = np.mean(permuse[17, 67, :])
            a2[k, 16] = np.mean(permuse[12, 66, :])
            a2[k, 17] = np.mean(permuse[37, 97, :])
            a2[k, 18] = np.mean(permuse[6, 63, :])
            a2[k, 19] = np.mean(permuse[14, 75, :])
            a2[k, 20] = np.mean(permuse[12, 66, :])
            a2[k, 21] = np.mean(permuse[10, 27, :])

            uniewater = water_use[k, :, :, :]
            permuse = uniewater
            a3[k, 0] = np.mean(permuse[14, 30, :])
            a3[k, 1] = np.mean(permuse[9, 31, :])
            a3[k, 2] = np.mean(permuse[13, 33, :])
            a3[k, 3] = np.mean(permuse[8, 36, :])
            a3[k, 4] = np.mean(permuse[8, 45, :])
            a3[k, 5] = np.mean(permuse[9, 28, :])
            a3[k, 6] = np.mean(permuse[9, 23, :])
            a3[k, 7] = np.mean(permuse[21, 21, :])
            a3[k, 8] = np.mean(permuse[13, 27, :])
            a3[k, 9] = np.mean(permuse[18, 37, :])
            a3[k, 10] = np.mean(permuse[18, 53, :])
            a3[k, 11] = np.mean(permuse[15, 65, :])
            a3[k, 12] = np.mean(permuse[24, 36, :])
            a3[k, 13] = np.mean(permuse[18, 54, :])
            a3[k, 14] = np.mean(permuse[11, 71, :])
            a3[k, 15] = np.mean(permuse[17, 67, :])
            a3[k, 16] = np.mean(permuse[12, 66, :])
            a3[k, 17] = np.mean(permuse[37, 97, :])
            a3[k, 18] = np.mean(permuse[6, 63, :])
            a3[k, 19] = np.mean(permuse[14, 75, :])
            a3[k, 20] = np.mean(permuse[12, 66, :])
            a3[k, 21] = np.mean(permuse[10, 27, :])

            unietime = Time_use[k, :, :, :]
            permuse = unietime
            a4[k, 0] = permuse[0, 0, 0]

        inn1 = np.hstack((permxx, a1, 1 - (a2 + a3), a2, a3, a4))

        innn[i, :, :] = inn1

        os.chdir(oldfolder)
    return innn, ouut


def Get_data_FFNN1(
    folder, oldfolder, N, pressure, Sgas, Swater, perm, Time, steppi, steppi_indices
):

    ouut = np.zeros((N, steppi, 66))
    innn = np.zeros((N, steppi, 90))
    steppii = 246
    for i in range(N):
        WOIL1 = np.zeros((steppi, 22))
        WWATER1 = np.zeros((steppi, 22))
        WGAS1 = np.zeros((steppi, 22))
        WWINJ1 = np.zeros((steppi, 9))
        WGASJ1 = np.zeros((steppi, 4))

        # f = 'Realisation'
        # folder = f + str(i)
        # folder = to_absolute_path('../RUNS/Realisation' + str(i))
        os.chdir(folder)
        ## IMPORT FOR OIL
        A2oilsim = pd.read_csv("FULLNORNE2.RSM", skiprows=1545, sep="\s+", header=None)

        B_1BHoilsim = A2oilsim[5].values[:steppii]
        B_1Hoilsim = A2oilsim[6].values[:steppii]
        B_2Hoilsim = A2oilsim[7].values[:steppii]
        B_3Hoilsim = A2oilsim[8].values[:steppii]
        B_4BHoilsim = A2oilsim[9].values[:steppii]

        A22oilsim = pd.read_csv("FULLNORNE2.RSM", skiprows=1801, sep="\s+", header=None)
        B_4DHoilsim = A22oilsim[1].values[:steppii]
        B_4Hoilsim = A22oilsim[2].values[:steppii]
        D_1CHoilsim = A22oilsim[3].values[:steppii]
        D_1Hoilsim = A22oilsim[4].values[:steppii]
        D_2Hoilsim = A22oilsim[5].values[:steppii]
        D_3AHoilsim = A22oilsim[6].values[:steppii]
        D_3BHoilsim = A22oilsim[7].values[:steppii]
        D_4AHoilsim = A22oilsim[8].values[:steppii]
        D_4Hoilsim = A22oilsim[9].values[:steppii]

        A222oilsim = pd.read_csv(
            "FULLNORNE2.RSM", skiprows=2057, sep="\s+", header=None
        )

        E_1Hoilsim = A222oilsim[1].values[:steppii]
        E_2AHoilsim = A222oilsim[2].values[:steppii]
        E_2Hoilsim = A222oilsim[3].values[:steppii]
        E_3AHoilsim = A222oilsim[4].values[:steppii]
        E_3CHoilsim = A222oilsim[5].values[:steppii]
        E_3Hoilsim = A222oilsim[6].values[:steppii]
        E_4AHoilsim = A222oilsim[7].values[:steppii]
        K_3Hoilsim = A222oilsim[8].values[:steppii]

        WOIL1[:, 0] = B_1BHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 1] = B_1Hoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 2] = B_2Hoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 3] = B_3Hoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 4] = B_4BHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 5] = B_4DHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 6] = B_4Hoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 7] = D_1CHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 8] = D_1Hoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 9] = D_2Hoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 10] = D_3AHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 11] = D_3BHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 12] = D_4AHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 13] = D_4Hoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 14] = E_1Hoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 15] = E_2AHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 16] = E_2Hoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 17] = E_3AHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 18] = E_3CHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 19] = E_3Hoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 20] = E_4AHoilsim.ravel()[steppi_indices - 1]
        WOIL1[:, 21] = K_3Hoilsim.ravel()[steppi_indices - 1]

        ##IMPORT FOR WATER
        A2watersim = pd.read_csv(
            "FULLNORNE2.RSM", skiprows=2313, sep="\s+", header=None
        )
        B_1BHwatersim = A2watersim[9].values[:steppii]

        A22watersim = pd.read_csv(
            "FULLNORNE2.RSM", skiprows=2569, sep="\s+", header=None
        )
        B_1Hwatersim = A22watersim[1].values[:steppii]
        B_2Hwatersim = A22watersim[2].values[:steppii]
        B_3Hwatersim = A22watersim[3].values[:steppii]
        B_4BHwatersim = A22watersim[4].values[:steppii]
        B_4DHwatersim = A22watersim[5].values[:steppii]
        B_4Hwatersim = A22watersim[6].values[:steppii]
        D_1CHwatersim = A22watersim[7].values[:steppii]
        D_1Hwatersim = A22watersim[8].values[:steppii]
        D_2Hwatersim = A22watersim[9].values[:steppii]

        A222watersim = pd.read_csv(
            "FULLNORNE2.RSM", skiprows=2825, sep="\s+", header=None
        )
        D_3AHwatersim = A222watersim[1].values[:steppii]
        D_3BHwatersim = A222watersim[2].values[:steppii]
        D_4AHwatersim = A222watersim[3].values[:steppii]
        D_4Hwatersim = A222watersim[4].values[:steppii]
        E_1Hwatersim = A222watersim[5].values[:steppii]
        E_2AHwatersim = A222watersim[6].values[:steppii]
        E_2Hwatersim = A222watersim[7].values[:steppii]
        E_3AHwatersim = A222watersim[8].values[:steppii]
        E_3CHwatersim = A222watersim[9].values[:steppii]

        A222watersim = pd.read_csv(
            "FULLNORNE2.RSM", skiprows=3081, sep="\s+", header=None
        )
        E_3Hwatersim = A222watersim[1].values[:steppii]
        E_4AHwatersim = A222watersim[2].values[:steppii]
        K_3Hwatersim = A222watersim[3].values[:steppii]

        WWATER1[:, 0] = B_1BHwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 1] = B_1Hwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 2] = B_2Hwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 3] = B_3Hwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 4] = B_4BHwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 5] = B_4DHwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 6] = B_4Hwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 7] = D_1CHwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 8] = D_1Hwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 9] = D_2Hwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 10] = D_3AHwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 11] = D_3BHwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 12] = D_4AHwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 13] = D_4Hwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 14] = E_1Hwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 15] = E_2AHwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 16] = E_2Hwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 17] = E_3AHwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 18] = E_3CHwatersim.ravel()[steppi_indices - 1]

        WWATER1[:, 19] = E_3Hwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 20] = E_4AHwatersim.ravel()[steppi_indices - 1]
        WWATER1[:, 21] = K_3Hwatersim.ravel()[steppi_indices - 1]

        ## GAS PRODUCTION RATE
        A2gassim = pd.read_csv("FULLNORNE2.RSM", skiprows=1033, sep="\s+", header=None)
        B_1BHgassim = A2gassim[1].values[:steppii]
        B_1Hgassim = A2gassim[2].values[:steppii]
        B_2Hgassim = A2gassim[3].values[:steppii]
        B_3Hgassim = A2gassim[4].values[:steppii]
        B_4BHgassim = A2gassim[5].values[:steppii]
        B_4DHgassim = A2gassim[6].values[:steppii]
        B_4Hgassim = A2gassim[7].values[:steppii]
        D_1CHgassim = A2gassim[8].values[:steppii]
        D_1Hgassim = A2gassim[9].values[:steppii]

        A22gassim = pd.read_csv("FULLNORNE2.RSM", skiprows=1289, sep="\s+", header=None)
        D_2Hgassim = A22gassim[1].values[:steppii]
        D_3AHgassim = A22gassim[2].values[:steppii]
        D_3BHgassim = A22gassim[3].values[:steppii]
        D_4AHgassim = A22gassim[4].values[:steppii]
        D_4Hgassim = A22gassim[5].values[:steppii]
        E_1Hgassim = A22gassim[6].values[:steppii]
        E_2AHgassim = A22gassim[7].values[:steppii]
        E_2Hgassim = A22gassim[8].values[:steppii]
        E_3AHgassim = A22gassim[9].values[:steppii]

        A222gassim = pd.read_csv(
            "FULLNORNE2.RSM", skiprows=1545, sep="\s+", header=None
        )
        E_3CHgassim = A222gassim[1].values[:steppii]
        E_3Hgassim = A222gassim[2].values[:steppii]
        E_4AHgassim = A222gassim[3].values[:steppii]
        K_3Hgassim = A222gassim[4].values[:steppii]

        WGAS1[:, 0] = B_1BHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 1] = B_1Hgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 2] = B_2Hgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 3] = B_3Hgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 4] = B_4BHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 5] = B_4DHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 6] = B_4Hgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 7] = D_1CHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 8] = D_1Hgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 9] = D_2Hgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 10] = D_3AHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 11] = D_3BHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 12] = D_4AHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 13] = D_4Hgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 14] = E_1Hgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 15] = E_2AHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 16] = E_2Hgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 17] = E_3AHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 18] = E_3CHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 19] = E_3Hgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 20] = E_4AHgassim.ravel()[steppi_indices - 1]
        WGAS1[:, 21] = K_3Hgassim.ravel()[steppi_indices - 1]

        ## WATER INJECTOR RATE
        A2wir = pd.read_csv("FULLNORNE2.RSM", skiprows=2057, sep="\s+", header=None)
        C_1Hwaterinjsim = A2wir[9].values[:steppii]

        A22wir = pd.read_csv("FULLNORNE2.RSM", skiprows=2313, sep="\s+", header=None)
        C_2Hwaterinjsim = A22wir[1].values[:steppii]
        C_3Hwaterinjsim = A22wir[2].values[:steppii]
        C_4AHwaterinjsim = A22wir[3].values[:steppii]
        C_4Hwaterinjsim = A22wir[4].values[:steppii]
        F_1Hwaterinjsim = A22wir[5].values[:steppii]
        F_2Hwaterinjsim = A22wir[6].values[:steppii]
        F_3Hwaterinjsim = A22wir[7].values[:steppii]
        F_4Hwaterinjsim = A22wir[8].values[:steppii]

        WWINJ1[:, 0] = C_1Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ1[:, 1] = C_2Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ1[:, 2] = C_3Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ1[:, 3] = C_4AHwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ1[:, 4] = C_4Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ1[:, 5] = F_1Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ1[:, 6] = F_2Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ1[:, 7] = F_3Hwaterinjsim.ravel()[steppi_indices - 1]
        WWINJ1[:, 8] = F_4Hwaterinjsim.ravel()[steppi_indices - 1]

        ## GAS INJECTOR RATE
        A2gir = pd.read_csv("FULLNORNE2.RSM", skiprows=777, sep="\s+", header=None)
        C_1Hgasinjsim = A2gir[6].values[:steppii]
        C_3Hgasinjsim = A2gir[7].values[:steppii]
        C_4AHgasinjsim = A2gir[8].values[:steppii]
        C_4Hgasinjsim = A2gir[9].values[:steppii]

        WGASJ1[:, 0] = C_1Hgasinjsim.ravel()[steppi_indices - 1]
        WGASJ1[:, 1] = C_3Hgasinjsim.ravel()[steppi_indices - 1]
        WGASJ1[:, 2] = C_4AHgasinjsim.ravel()[steppi_indices - 1]
        WGASJ1[:, 3] = C_4Hgasinjsim.ravel()[steppi_indices - 1]

        # Return to the root folder
        out = np.hstack((WOIL1, WWATER1, WGAS1))
        out[out <= 0] = 0
        ouut[i, :, :] = out

        ##INPUTS

        permuse = perm[i, 0, :, :, :]
        a1 = np.mean(permuse[14, 30, :]) * np.ones((steppi, 1))
        a2 = np.mean(permuse[9, 31, :]) * np.ones((steppi, 1))
        a3 = np.mean(permuse[13, 33, :]) * np.ones((steppi, 1))
        a5 = np.mean(permuse[8, 36, :]) * np.ones((steppi, 1))
        a6 = np.mean(permuse[8, 45, :]) * np.ones((steppi, 1))
        a7 = np.mean(permuse[9, 28, :]) * np.ones((steppi, 1))
        a8 = np.mean(permuse[9, 23, :]) * np.ones((steppi, 1))
        a9 = np.mean(permuse[21, 21, :]) * np.ones((steppi, 1))
        a10 = np.mean(permuse[13, 27, :]) * np.ones((steppi, 1))
        a11 = np.mean(permuse[18, 37, :]) * np.ones((steppi, 1))
        a12 = np.mean(permuse[18, 53, :]) * np.ones((steppi, 1))
        a13 = np.mean(permuse[15, 65, :]) * np.ones((steppi, 1))
        a14 = np.mean(permuse[24, 36, :]) * np.ones((steppi, 1))
        a15 = np.mean(permuse[18, 53, :]) * np.ones((steppi, 1))
        a16 = np.mean(permuse[11, 71, :]) * np.ones((steppi, 1))
        a17 = np.mean(permuse[17, 67, :]) * np.ones((steppi, 1))
        a18 = np.mean(permuse[12, 66, :]) * np.ones((steppi, 1))
        a19 = np.mean(permuse[37, 97, :]) * np.ones((steppi, 1))
        a20 = np.mean(permuse[6, 63, :]) * np.ones((steppi, 1))
        a21 = np.mean(permuse[14, 75, :]) * np.ones((steppi, 1))
        a22 = np.mean(permuse[12, 66, :]) * np.ones((steppi, 1))
        a23 = np.mean(permuse[10, 27, :]) * np.ones((steppi, 1))

        permxx = np.hstack(
            (
                a1,
                a2,
                a3,
                a5,
                a6,
                a7,
                a8,
                a9,
                a10,
                a11,
                a12,
                a13,
                a14,
                a15,
                a16,
                a17,
                a18,
                a19,
                a20,
                a21,
                a22,
                a23,
            )
        )

        presure_use = pressure[0, :, :, :, :]
        gas_use = Sgas[0, :, :, :, :]
        water_use = Swater[0, :, :, :, :]
        Time_use = Time[0, :, :, :, :]

        a1 = np.zeros((steppi, 1))
        a2 = np.zeros((steppi, 22))
        a3 = np.zeros((steppi, 22))
        a4 = np.zeros((steppi, 1))

        for k in range(steppi):
            uniep = presure_use[k, :, :, :]
            permuse = uniep
            a1[k, 0] = np.mean(permuse)

            uniegas = gas_use[k, :, :, :]
            permuse = uniegas
            a2[k, 0] = np.mean(permuse[14, 30, :])
            a2[k, 1] = np.mean(permuse[9, 31, :])
            a2[k, 2] = np.mean(permuse[13, 33, :])
            a2[k, 3] = np.mean(permuse[8, 36, :])
            a2[k, 4] = np.mean(permuse[8, 45, :])
            a2[k, 5] = np.mean(permuse[9, 28, :])
            a2[k, 6] = np.mean(permuse[9, 23, :])
            a2[k, 7] = np.mean(permuse[21, 21, :])
            a2[k, 8] = np.mean(permuse[13, 27, :])
            a2[k, 9] = np.mean(permuse[18, 37, :])
            a2[k, 10] = np.mean(permuse[18, 53, :])
            a2[k, 11] = np.mean(permuse[15, 65, :])
            a2[k, 12] = np.mean(permuse[24, 36, :])
            a2[k, 13] = np.mean(permuse[18, 54, :])
            a2[k, 14] = np.mean(permuse[11, 71, :])
            a2[k, 15] = np.mean(permuse[17, 67, :])
            a2[k, 16] = np.mean(permuse[12, 66, :])
            a2[k, 17] = np.mean(permuse[37, 97, :])
            a2[k, 18] = np.mean(permuse[6, 63, :])
            a2[k, 19] = np.mean(permuse[14, 75, :])
            a2[k, 20] = np.mean(permuse[12, 66, :])
            a2[k, 21] = np.mean(permuse[10, 27, :])

            uniewater = water_use[k, :, :, :]
            permuse = uniewater
            a3[k, 0] = np.mean(permuse[14, 30, :])
            a3[k, 1] = np.mean(permuse[9, 31, :])
            a3[k, 2] = np.mean(permuse[13, 33, :])
            a3[k, 3] = np.mean(permuse[8, 36, :])
            a3[k, 4] = np.mean(permuse[8, 45, :])
            a3[k, 5] = np.mean(permuse[9, 28, :])
            a3[k, 6] = np.mean(permuse[9, 23, :])
            a3[k, 7] = np.mean(permuse[21, 21, :])
            a3[k, 8] = np.mean(permuse[13, 27, :])
            a3[k, 9] = np.mean(permuse[18, 37, :])
            a3[k, 10] = np.mean(permuse[18, 53, :])
            a3[k, 11] = np.mean(permuse[15, 65, :])
            a3[k, 12] = np.mean(permuse[24, 36, :])
            a3[k, 13] = np.mean(permuse[18, 54, :])
            a3[k, 14] = np.mean(permuse[11, 71, :])
            a3[k, 15] = np.mean(permuse[17, 67, :])
            a3[k, 16] = np.mean(permuse[12, 66, :])
            a3[k, 17] = np.mean(permuse[37, 97, :])
            a3[k, 18] = np.mean(permuse[6, 63, :])
            a3[k, 19] = np.mean(permuse[14, 75, :])
            a3[k, 20] = np.mean(permuse[12, 66, :])
            a3[k, 21] = np.mean(permuse[10, 27, :])

            unietime = Time_use[k, :, :, :]
            permuse = unietime
            a4[k, 0] = permuse[0, 0, 0]

        inn1 = np.hstack((permxx, a1, 1 - (a2 + a3), a2, a3, a4))

        innn[0, :, :] = inn1

        os.chdir(oldfolder)
    return innn, ouut


def Remove_folder(N_ens, straa):
    for jj in range(N_ens):
        folderr = straa + str(jj)
        rmtree(folderr)


def historydata(timestep, steppi, steppi_indices):
    WOIL1 = np.zeros((steppi, 22))
    WWATER1 = np.zeros((steppi, 22))
    WGAS1 = np.zeros((steppi, 22))
    WWINJ1 = np.zeros((steppi, 9))
    WGASJ1 = np.zeros((steppi, 4))

    indices = timestep
    print(" Get the Well Oil Production Rate")

    # Open the file and read lines until '---' is found
    lines = []
    with open("../NORNE/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 47873:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)

    # Now convert the lines into a DataFrame
    df = pd.DataFrame([l.split() for l in lines])

    # Convert columns types
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A1 = df[[2, 3, 4, 5, 6, 8]].values

    B_2H = A1[:, 0][indices - 1]
    D_1H = A1[:, 1][indices - 1]
    D_2H = A1[:, 2][indices - 1]
    B_4H = A1[:, 3][indices - 1]
    D_4H = A1[:, 4][indices - 1]
    E_3H = A1[:, 5][indices - 1]

    # A2 = np.genfromtxt('NORNE_ATW2013.RSM', dtype=float, usecols=(1, 2, 5, 6, 8, 9), skip_header=48743)

    # Open the file and read lines until '---' is found
    lines = []
    with open("../NORNE/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 48743:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)

    # Now convert the lines into a DataFrame
    df = pd.DataFrame([l.split() for l in lines])

    # Convert columns types
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A2 = df[[1, 4, 5, 7, 9]].values

    B_1H = A2[:, 0][indices - 1]
    B_3H = A2[:, 1][indices - 1]
    E_1H = A2[:, 2][indices - 1]
    E_2H = A2[:, 3][indices - 1]
    E_4AH = A2[:, 4][indices - 1]

    # Open the file and read lines until '---' is found
    lines = []
    with open("../NORNE/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 49613:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)

    # Now convert the lines into a DataFrame
    df = pd.DataFrame([l.split() for l in lines])

    # Convert columns types
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A3 = df[[2, 4, 7, 8, 9]].values

    D_3AH = A3[:, 0][indices - 1]
    E_3AH = A3[:, 1][indices - 1]
    B_4BH = A3[:, 2][indices - 1]
    D_4AH = A3[:, 3][indices - 1]
    D_1CH = A3[:, 4][indices - 1]

    # A4 = np.genfromtxt('NORNE_ATW2013.RSM', dtype=float, usecols=(2, 5, 6, 7, 8, 9), skip_header=50483)

    # Open the file and read lines until '---' is found
    lines = []
    with open("../NORNE/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 50483:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)

    # Now convert the lines into a DataFrame
    df = pd.DataFrame([l.split() for l in lines])

    # Convert columns types
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A4 = df[[2, 4, 5, 6, 8, 9]].values

    B_4DH = A4[:, 0][indices - 1]
    E_3CH = A4[:, 1][indices - 1]
    E_2AH = A4[:, 2][indices - 1]
    D_3BH = A4[:, 3][indices - 1]
    B_1BH = A4[:, 4][indices - 1]
    K_3H = A4[:, 5][indices - 1]

    WOIL1[:, 0] = B_1BH.ravel()[steppi_indices - 1]
    WOIL1[:, 1] = B_1H.ravel()[steppi_indices - 1]
    WOIL1[:, 2] = B_2H.ravel()[steppi_indices - 1]
    WOIL1[:, 3] = B_3H.ravel()[steppi_indices - 1]
    WOIL1[:, 4] = B_4BH.ravel()[steppi_indices - 1]
    WOIL1[:, 5] = B_4DH.ravel()[steppi_indices - 1]
    WOIL1[:, 6] = B_4H.ravel()[steppi_indices - 1]
    WOIL1[:, 7] = D_1CH.ravel()[steppi_indices - 1]
    WOIL1[:, 8] = D_1H.ravel()[steppi_indices - 1]
    WOIL1[:, 9] = D_2H.ravel()[steppi_indices - 1]
    WOIL1[:, 10] = D_3AH.ravel()[steppi_indices - 1]
    WOIL1[:, 11] = D_3BH.ravel()[steppi_indices - 1]
    WOIL1[:, 12] = D_4AH.ravel()[steppi_indices - 1]
    WOIL1[:, 13] = D_4H.ravel()[steppi_indices - 1]
    WOIL1[:, 14] = E_1H.ravel()[steppi_indices - 1]
    WOIL1[:, 15] = E_2AH.ravel()[steppi_indices - 1]
    WOIL1[:, 16] = E_2H.ravel()[steppi_indices - 1]
    WOIL1[:, 17] = E_3AH.ravel()[steppi_indices - 1]
    WOIL1[:, 18] = E_3CH.ravel()[steppi_indices - 1]
    WOIL1[:, 19] = E_3H.ravel()[steppi_indices - 1]
    WOIL1[:, 20] = E_4AH.ravel()[steppi_indices - 1]
    WOIL1[:, 21] = K_3H.ravel()[steppi_indices - 1]

    # Data for Water production rate
    print(" Get the Well water Production Rate")
    # A1w = np.genfromtxt('NORNE_ATW2013.RSM', dtype=float, usecols=(2, 3, 4, 5, 6, 8), skip_header=40913)
    # Open the file and read lines until '---' is found
    lines = []
    with open("../NORNE/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 40913:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)

    # Now convert the lines into a DataFrame
    df = pd.DataFrame([l.split() for l in lines])

    # Convert columns types
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A1w = df[[2, 3, 4, 5, 6, 8]].values

    B_2Hw = A1w[:, 0][indices - 1]
    D_1Hw = A1w[:, 1][indices - 1]
    D_2Hw = A1w[:, 2][indices - 1]
    B_4Hw = A1w[:, 3][indices - 1]
    D_4Hw = A1w[:, 4][indices - 1]
    E_3Hw = A1w[:, 5][indices - 1]

    # A2w = np.genfromtxt('NORNE_ATW2013.RSM', dtype=float, usecols=(1, 2, 5, 6, 8, 9), skip_header=48743)

    # Open the file and read lines until '---' is found
    lines = []
    with open("../NORNE/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 41783:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)

    # Now convert the lines into a DataFrame
    df = pd.DataFrame([l.split() for l in lines])

    # Convert columns types
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A2w = df[[1, 4, 5, 7, 9]].values

    B_1Hw = A2w[:, 0][indices - 1]
    B_3Hw = A2w[:, 1][indices - 1]
    E_1Hw = A2w[:, 2][indices - 1]
    E_2Hw = A2w[:, 3][indices - 1]
    E_4AHw = A2w[:, 4][indices - 1]

    # A3w = np.genfromtxt('NORNE_ATW2013.RSM', dtype=float, usecols=(2, 4, 7, 8, 9), skip_header=49613)
    # Open the file and read lines until '---' is found
    lines = []
    with open("../NORNE/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 42653:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)

    # Now convert the lines into a DataFrame
    df = pd.DataFrame([l.split() for l in lines])

    # Convert columns types
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A3w = df[[2, 4, 7, 8, 9]].values

    D_3AHw = A3w[:, 0][indices - 1]
    E_3AHw = A3w[:, 1][indices - 1]
    B_4BHw = A3w[:, 2][indices - 1]
    D_4AHw = A3w[:, 3][indices - 1]
    D_1CHw = A3w[:, 4][indices - 1]

    # A4w = np.genfromtxt('NORNE_ATW2013.RSM', dtype=float, usecols=(2, 5, 6, 7, 8, 9), skip_header=50483)

    # Open the file and read lines until '---' is found
    lines = []
    with open("../NORNE/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 43523:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)

    # Now convert the lines into a DataFrame
    df = pd.DataFrame([l.split() for l in lines])

    # Convert columns types
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A4w = df[[2, 4, 5, 6, 8, 9]].values

    B_4DHw = A4w[:, 0][indices - 1]
    E_3CHw = A4w[:, 1][indices - 1]
    E_2AHw = A4w[:, 2][indices - 1]
    D_3BHw = A4w[:, 3][indices - 1]
    B_1BHw = A4w[:, 4][indices - 1]
    K_3Hw = A4w[:, 5][indices - 1]

    WWATER1[:, 0] = B_1BHw.ravel()[steppi_indices - 1]
    WWATER1[:, 1] = B_1Hw.ravel()[steppi_indices - 1]
    WWATER1[:, 2] = B_2Hw.ravel()[steppi_indices - 1]
    WWATER1[:, 3] = B_3Hw.ravel()[steppi_indices - 1]
    WWATER1[:, 4] = B_4BHw.ravel()[steppi_indices - 1]
    WWATER1[:, 5] = B_4DHw.ravel()[steppi_indices - 1]
    WWATER1[:, 6] = B_4Hw.ravel()[steppi_indices - 1]
    WWATER1[:, 7] = D_1CHw.ravel()[steppi_indices - 1]
    WWATER1[:, 8] = D_1Hw.ravel()[steppi_indices - 1]
    WWATER1[:, 9] = D_2Hw.ravel()[steppi_indices - 1]
    WWATER1[:, 10] = D_3AHw.ravel()[steppi_indices - 1]
    WWATER1[:, 11] = D_3BHw.ravel()[steppi_indices - 1]
    WWATER1[:, 12] = D_4AHw.ravel()[steppi_indices - 1]
    WWATER1[:, 13] = D_4Hw.ravel()[steppi_indices - 1]
    WWATER1[:, 14] = E_1Hw.ravel()[steppi_indices - 1]
    WWATER1[:, 15] = E_2AHw.ravel()[steppi_indices - 1]
    WWATER1[:, 16] = E_2Hw.ravel()[steppi_indices - 1]
    WWATER1[:, 17] = E_3AHw.ravel()[steppi_indices - 1]
    WWATER1[:, 18] = E_3CHw.ravel()[steppi_indices - 1]

    WWATER1[:, 19] = E_3Hw.ravel()[steppi_indices - 1]
    WWATER1[:, 20] = E_4AHw.ravel()[steppi_indices - 1]
    WWATER1[:, 21] = K_3Hw.ravel()[steppi_indices - 1]

    # Data for Gas production rate
    print(" Get the Well Gas Production Rate")
    # A1g = np.genfromtxt('NORNE_ATW2013.RSM', dtype=float, usecols=(2, 3, 4, 5, 6, 8), skip_header=54833)
    lines = []
    with open("../NORNE/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 54833:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)

    # Now convert the lines into a DataFrame
    df = pd.DataFrame([l.split() for l in lines])

    # Convert columns types
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A1g = df[[2, 3, 4, 5, 6, 8]].values

    B_2Hg = A1g[:, 0][indices - 1]
    D_1Hg = A1g[:, 1][indices - 1]
    D_2Hg = A1g[:, 2][indices - 1]
    B_4Hg = A1g[:, 3][indices - 1]
    D_4Hg = A1g[:, 4][indices - 1]
    E_3Hg = A1g[:, 5][indices - 1]

    # A2g = np.genfromtxt('NORNE_ATW2013.RSM', dtype=float, usecols=(1, 2, 5, 6, 8, 9), skip_header=55703)

    # Open the file and read lines until '---' is found
    lines = []
    with open("../NORNE/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 55703:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)

    # Now convert the lines into a DataFrame
    df = pd.DataFrame([l.split() for l in lines])

    # Convert columns types
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A2g = df[[1, 4, 5, 7, 9]].values

    B_1Hg = A2g[:, 0][indices - 1]
    B_3Hg = A2g[:, 1][indices - 1]
    E_1Hg = A2g[:, 2][indices - 1]
    E_2Hg = A2g[:, 3][indices - 1]
    E_4AHg = A2g[:, 4][indices - 1]

    # A3g = np.genfromtxt('NORNE_ATW2013.RSM', dtype=float, usecols=(2, 4, 7, 8, 9), skip_header=56573)

    lines = []
    with open("../NORNE/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 56573:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)

    # Now convert the lines into a DataFrame
    df = pd.DataFrame([l.split() for l in lines])

    # Convert columns types
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A3g = df[[2, 4, 7, 8, 9]].values

    D_3AHg = A3g[:, 0][indices - 1]
    E_3AHg = A3g[:, 1][indices - 1]
    B_4BHg = A3g[:, 2][indices - 1]
    D_4AHg = A3g[:, 3][indices - 1]
    D_1CHg = A3g[:, 4][indices - 1]

    # A4g = np.genfromtxt('NORNE_ATW2013.RSM', dtype=float, usecols=(2, 5, 6, 7, 8, 9), skip_header=57443)
    # Open the file and read lines until '---' is found
    lines = []
    with open("../NORNE/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 57443:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)

    # Now convert the lines into a DataFrame
    df = pd.DataFrame([l.split() for l in lines])

    # Convert columns types
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A4g = df[[2, 4, 5, 6, 8, 9]].values
    B_4DHg = A4g[:, 0][indices - 1]
    E_3CHg = A4g[:, 1][indices - 1]
    E_2AHg = A4g[:, 2][indices - 1]
    D_3BHg = A4g[:, 3][indices - 1]
    B_1BHg = A4g[:, 4][indices - 1]
    K_3Hg = A4g[:, 5][indices - 1]

    WGAS1[:, 0] = B_1BHg.ravel()[steppi_indices - 1]
    WGAS1[:, 1] = B_1Hg.ravel()[steppi_indices - 1]
    WGAS1[:, 2] = B_2Hg.ravel()[steppi_indices - 1]
    WGAS1[:, 3] = B_3Hg.ravel()[steppi_indices - 1]
    WGAS1[:, 4] = B_4BHg.ravel()[steppi_indices - 1]
    WGAS1[:, 5] = B_4DHg.ravel()[steppi_indices - 1]
    WGAS1[:, 6] = B_4Hg.ravel()[steppi_indices - 1]
    WGAS1[:, 7] = D_1CHg.ravel()[steppi_indices - 1]
    WGAS1[:, 8] = D_1Hg.ravel()[steppi_indices - 1]
    WGAS1[:, 9] = D_2Hg.ravel()[steppi_indices - 1]
    WGAS1[:, 10] = D_3AHg.ravel()[steppi_indices - 1]
    WGAS1[:, 11] = D_3BHg.ravel()[steppi_indices - 1]
    WGAS1[:, 12] = D_4AHg.ravel()[steppi_indices - 1]
    WGAS1[:, 13] = D_4Hg.ravel()[steppi_indices - 1]
    WGAS1[:, 14] = E_1Hg.ravel()[steppi_indices - 1]
    WGAS1[:, 15] = E_2AHg.ravel()[steppi_indices - 1]
    WGAS1[:, 16] = E_2Hg.ravel()[steppi_indices - 1]
    WGAS1[:, 17] = E_3AHg.ravel()[steppi_indices - 1]
    WGAS1[:, 18] = E_3CHg.ravel()[steppi_indices - 1]
    WGAS1[:, 19] = E_3Hg.ravel()[steppi_indices - 1]
    WGAS1[:, 20] = E_4AHg.ravel()[steppi_indices - 1]
    WGAS1[:, 21] = K_3Hg.ravel()[steppi_indices - 1]

    # Data for Water injection rate
    print(" Get the Well water injection Rate")
    # A1win = np.genfromtxt('NORNE_ATW2013.RSM', dtype=float, usecols=(1, 2, 3, 4, 5, 7), skip_header=72237)

    # Open the file and read lines until '---' is found
    lines = []
    with open("../NONRE/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 72237:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)

    # Now convert the lines into a DataFrame
    df = pd.DataFrame([l.split() for l in lines])

    # Convert columns types
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A1win = df[[1, 2, 3, 4, 5, 6, 7, 8, 9]].values

    C_1Hwin = A1win[:, 0][indices - 1]
    C_2Hwin = A1win[:, 1][indices - 1]
    C_3Hwin = A1win[:, 2][indices - 1]
    C_4Hwin = A1win[:, 3][indices - 1]
    C_4AHwin = A1win[:, 4][indices - 1]
    F_1Hwin = A1win[:, 5][indices - 1]
    F_2Hwin = A1win[:, 6][indices - 1]
    F_3Hwin = A1win[:, 7][indices - 1]
    F_4Hwin = A1win[:, 8][indices - 1]

    WWINJ1[:, 0] = C_1Hwin.ravel()[steppi_indices - 1]
    WWINJ1[:, 1] = C_2Hwin.ravel()[steppi_indices - 1]
    WWINJ1[:, 2] = C_3Hwin.ravel()[steppi_indices - 1]
    WWINJ1[:, 3] = C_4AHwin.ravel()[steppi_indices - 1]
    WWINJ1[:, 4] = C_4Hwin.ravel()[steppi_indices - 1]
    WWINJ1[:, 5] = F_1Hwin.ravel()[steppi_indices - 1]
    WWINJ1[:, 6] = F_2Hwin.ravel()[steppi_indices - 1]
    WWINJ1[:, 7] = F_3Hwin.ravel()[steppi_indices - 1]
    WWINJ1[:, 8] = F_4Hwin.ravel()[steppi_indices - 1]

    # Data for Gas injection rate
    print(" Get the Well Gas injection Rate")
    # A1gin = np.genfromtxt('NORNE_ATW2013.RSM', dtype=float, usecols=(1, 3, 4, 5, 6, 8), skip_header=73977)
    # Open the file and read lines until '---' is found
    lines = []
    with open("../NORNE/NORNE_ATW2013.RSM", "r") as f:
        for i, line in enumerate(f):
            if i < 73977:  # Skip the first 47873 lines
                continue
            if "---" in line:  # Stop reading when '---' is found
                break
            lines.append(line)

    # Now convert the lines into a DataFrame
    df = pd.DataFrame([l.split() for l in lines])

    # Convert columns types
    df[0] = df[0].astype(str)
    for i in range(1, len(df.columns)):
        df[i] = df[i].astype(float)
    df.drop(df.index[-1], inplace=True)
    A1gin = df[[1, 3, 4, 5]].values
    C_1Hgin = A1gin[:, 0][indices - 1]
    C_3Hgin = A1gin[:, 1][indices - 1]
    C_4Hgin = A1gin[:, 2][indices - 1]
    C_4AHgin = A1gin[:, 3][indices - 1]

    WGASJ1[:, 0] = C_1Hgin.ravel()[steppi_indices - 1]
    WGASJ1[:, 1] = C_3Hgin.ravel()[steppi_indices - 1]
    WGASJ1[:, 2] = C_4AHgin.ravel()[steppi_indices - 1]
    WGASJ1[:, 3] = C_4Hgin.ravel()[steppi_indices - 1]

    # Get Data for plotting

    DATA = {
        "OIL": WOIL1,
        "WATER": WWATER1,
        "GAS": WGAS1,
        "WATER_INJ": WWINJ1,
        "WGAS_inj": WGASJ1,
    }

    oil = np.reshape(WOIL1, (-1, 1), "F")
    water = np.reshape(WWATER1, (-1, 1), "F")
    gas = np.reshape(WGAS1, (-1, 1), "F")
    winj = np.reshape(WWINJ1, (-1, 1), "F")
    gasinj = np.reshape(WGASJ1, (-1, 1), "F")

    # Get data for history matching
    DATA2 = np.vstack([oil, water, gas, winj, gasinj])
    return DATA, DATA2


# def linear_interp(x, xp, fp):
#     #left_indices = paddle.clamp(paddle.searchsorted(xp, x) - 1, 0, len(xp) - 2)
#     contiguous_xp = xp.contiguous()
#     left_indices = paddle.clamp(paddle.searchsorted(contiguous_xp, x) - 1, 0, len(contiguous_xp) - 2)
#     interpolated_value = (((fp[left_indices + 1] - fp[left_indices]) / (contiguous_xp[left_indices + 1] - contiguous_xp[left_indices])) \
#                           * (x - contiguous_xp[left_indices])) + fp[left_indices]
#     return interpolated_value


def linear_interp(x, xp, fp):
    contiguous_xp = xp.contiguous()
    left_indices = paddle.clamp(
        paddle.searchsorted(contiguous_xp, x) - 1, 0, len(contiguous_xp) - 2
    )

    # Calculate denominators and handle zero case
    denominators = contiguous_xp[left_indices + 1] - contiguous_xp[left_indices]
    close_to_zero = denominators.abs() < 1e-10
    denominators[close_to_zero] = 1.0  # or any non-zero value to avoid NaN

    interpolated_value = (
        ((fp[left_indices + 1] - fp[left_indices]) / denominators)
        * (x - contiguous_xp[left_indices])
    ) + fp[left_indices]
    return interpolated_value


def replace_nan_with_zero(tensor):
    nan_mask = paddle.isnan(tensor)
    return tensor * (~nan_mask).float() + nan_mask.float() * 0.0


def interp_torch(cuda, reference_matrix1, reference_matrix2, tensor1):
    chunk_size = 1

    chunks = paddle.chunk(tensor1, chunks=chunk_size, dim=0)
    processed_chunks = []
    for start_idx in range(chunk_size):
        interpolated_chunk = linear_interp(
            chunks[start_idx], reference_matrix1, reference_matrix2
        )
        processed_chunks.append(interpolated_chunk)

    paddle.device.cuda.empty_cache()
    return processed_chunks


def RelPerm(Sa, Sg, SWI, SWR, SWOW, SWOG):
    one_minus_swi_swr = 1 - (SWI + SWR)

    so = (((1 - (Sa + Sg))) - SWR) / one_minus_swi_swr
    sw = (Sa - SWI) / one_minus_swi_swr
    sg = Sg / one_minus_swi_swr

    KROW = linear_interp(Sa, SWOW[:, 0], SWOW[:, 1])
    KRW = linear_interp(Sa, SWOW[:, 0], SWOW[:, 2])
    KROG = linear_interp(Sg, SWOG[:, 0], SWOG[:, 1])
    KRG = linear_interp(Sg, SWOG[:, 0], SWOG[:, 2])

    KRO = ((KROW / (1 - sw)) * (KROG / (1 - sg))) * so
    return KRW, KRO, KRG


def write_RSM(data, Time, Name):
    # Generate a random numpy array of shape 100x66 for demonstration
    # data = np.random.rand(100, 66)

    # Create groups and individual column labels
    groups = ["WOPR(bbl/day)", "WWPR(bbl/day)", "WGPR(scf/day)"]
    columns = [
        "B_1BH",
        "B_1H",
        "B_2H",
        "B_3H",
        "B_4BH",
        "B_4DH",
        "B_4H",
        "D_1CH",
        "D_1H",
        "D_2H",
        "D_3AH",
        "D_3BH",
        "D_4AH",
        "D_4H",
        "E_1H",
        "E_2AH",
        "E_2H",
        "E_3AH",
        "E_3CH",
        "E_3H",
        "E_4AH",
        "K_3H",
    ]

    # Generate hierarchical column headers
    headers = pd.MultiIndex.from_product([groups, columns])

    # Create a Pandas DataFrame from the numpy array with the specified column headers
    df = pd.DataFrame(data, columns=headers)
    df.insert(0, "Time(days)", Time)

    # Create a Pandas Excel writer using XlsxWriter as the engine
    with pd.ExcelWriter(Name + ".xlsx", engine="xlsxwriter") as writer:
        workbook = writer.book

        # Add a new worksheet
        worksheet = workbook.add_worksheet("Sheet1")
        writer.sheets["Sheet1"] = worksheet

        # Define the format for the header: bold and centered
        header_format = workbook.add_format({"bold": True, "align": "center"})

        # Write the 'Time(days)' header with the format
        worksheet.write(0, 0, "Time(days)", header_format)

        # Write the second-level column headers
        col = 1
        for sub_col in columns * len(groups):
            worksheet.write(1, col, sub_col)
            col += 1

        # Manually merge cells for each top-level column label and apply the format
        col = 1
        for group in groups:
            end_col = col + len(columns) - 1
            worksheet.merge_range(0, col, 0, end_col, group, header_format)
            col = end_col + 1

        # Write the data rows, starting from row 2 (which is the third row in zero-based index)
        # to ensure there is no empty row between the second-level column headers and the data
        for row_num, row_data in enumerate(df.values):
            worksheet.write_row(row_num + 2, 0, row_data)

        # Write 'Time(days)' column
        time_data = np.arange(1, 101)
        for row_num, time_val in enumerate(time_data):
            worksheet.write(row_num + 2, 0, time_val)

        # Set the column width for better visibility (optional)
        worksheet.set_column(0, 0, 12)  # 'Time(days)' column
        worksheet.set_column(1, col, 10)  # Data columns


def compute_metrics(y_true, y_pred):
    y_true_mean = np.mean(y_true)
    TSS = np.sum((y_true - y_true_mean) ** 2)
    RSS = np.sum((y_true - y_pred) ** 2)

    R2 = 1 - (RSS / TSS)
    L2_accuracy = 1 - np.sqrt(RSS) / np.sqrt(TSS)

    return R2, L2_accuracy


def Plot_Modulus(
    ax, nx, ny, nz, Truee, N_injw, N_pr, N_injg, varii, injectors, producers, gass
):
    # matplotlib.use('Agg')
    Pressz = np.reshape(Truee, (nx, ny, nz), "F")

    avg_2d = np.mean(Pressz, axis=2)

    avg_2d[avg_2d == 0] = np.nan  # Convert zeros to NaNs

    maxii = max(Pressz.ravel())
    minii = min(Pressz.ravel())
    Pressz = Pressz / maxii

    masked_Pressz = Pressz
    colors = plt.cm.jet(masked_Pressz)
    colors[np.isnan(Pressz), :3] = 1  # set color to white for NaN values
    # alpha = np.where(np.isnan(Pressz), 0.0, 0.8) # set alpha to 0 for NaN values
    norm = mpl.colors.Normalize(vmin=minii, vmax=maxii)

    arr_3d = Pressz
    # fig = plt.figure(figsize=(20, 20), dpi = 200)
    # ax = fig.add_subplot(221, projection='3d')

    # Shift the coordinates to center the points at the voxel locations
    x, y, z = np.indices((arr_3d.shape))
    x = x + 0.5
    y = y + 0.5
    z = z + 0.5

    # Set the colors of each voxel using a jet colormap
    # colors = plt.cm.jet(arr_3d)
    # norm = matplotlib.colors.Normalize(vmin=minii, vmax=maxii)

    # Plot each voxel and save the mappable object
    ax.voxels(arr_3d, facecolors=colors, alpha=0.5, edgecolor="none", shade=True)
    m = cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
    m.set_array([])

    # Add a colorbar for the mappable object
    # plt.colorbar(mappable)
    # Set the axis labels and title
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    # ax.set_title(titti,fontsize= 14)

    # Set axis limits to reflect the extent of each axis of the matrix
    ax.set_xlim(0, arr_3d.shape[0])
    ax.set_ylim(0, arr_3d.shape[1])
    ax.set_zlim(0, arr_3d.shape[2])
    # ax.set_zlim(0, 60)

    # Remove the grid
    ax.grid(False)

    # Set lighting to bright
    # ax.set_facecolor('white')
    # Set the aspect ratio of the plot

    ax.set_box_aspect([nx, ny, nz])

    # Set the projection type to orthogonal
    ax.set_proj_type("ortho")

    # Remove the tick labels on each axis
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # Remove the tick lines on each axis
    ax.xaxis._axinfo["tick"]["inward_factor"] = 0
    ax.xaxis._axinfo["tick"]["outward_factor"] = 0.4
    ax.yaxis._axinfo["tick"]["inward_factor"] = 0
    ax.yaxis._axinfo["tick"]["outward_factor"] = 0.4
    ax.zaxis._axinfo["tick"]["inward_factor"] = 0
    ax.zaxis._axinfo["tick"]["outward_factor"] = 0.4
    # Set the azimuth and elevation to make the plot brighter
    ax.view_init(elev=30, azim=45)

    n_inj = N_injw  # Number of injectors
    n_prod = N_pr  # Number of producers

    for mm in range(n_inj):
        usethis = injectors[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[-1])
        # Define the direction of the line
        line_dir = (0, 0, (nz * 2) + 7)
        # Define the coordinates of the line end
        x_line_end = xloc + line_dir[0]
        y_line_end = yloc + line_dir[1]
        z_line_end = 0 + line_dir[2]
        ax.plot([xloc, xloc], [yloc, yloc], [0, (nz * 2) + 7], "blue", linewidth=1)
        ax.text(
            x_line_end,
            y_line_end,
            z_line_end,
            discrip,
            color="blue",
            weight="bold",
            fontsize=5,
        )

    for mm in range(n_prod):
        usethis = producers[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[-1])
        # Define the direction of the line
        line_dir = (0, 0, (nz * 2) + 5)
        # Define the coordinates of the line end
        x_line_end = xloc + line_dir[0]
        y_line_end = yloc + line_dir[1]
        z_line_end = 0 + line_dir[2]
        ax.plot([xloc, xloc], [yloc, yloc], [0, (nz * 2) + 5], "g", linewidth=1)
        ax.text(
            x_line_end,
            y_line_end,
            z_line_end,
            discrip,
            color="g",
            weight="bold",
            fontsize=5,
        )

    for mm in range(N_injg):
        usethis = gass[mm]
        xloc = int(usethis[0])
        yloc = int(usethis[1])
        discrip = str(usethis[-1])
        # Define the direction of the line
        line_dir = (0, 0, (nz * 2) + 5)
        # Define the coordinates of the line end
        x_line_end = xloc + line_dir[0]
        y_line_end = yloc + line_dir[1]
        z_line_end = 0 + line_dir[2]
        ax.plot([xloc, xloc], [yloc, yloc], [0, (nz * 2) + 2], "red", linewidth=1)
        ax.text(
            x_line_end,
            y_line_end,
            z_line_end,
            discrip,
            color="red",
            weight="bold",
            fontsize=5,
        )

    blue_line = mlines.Line2D([], [], color="blue", linewidth=2, label="water injector")
    green_line = mlines.Line2D(
        [], [], color="green", linewidth=2, label="oil/water/gas producer"
    )
    red_line = mlines.Line2D([], [], color="red", linewidth=2, label="gas injectors")

    # Add the legend to the plot
    ax.legend(handles=[blue_line, green_line, red_line], loc="lower left", fontsize=9)

    # Add a horizontal colorbar to the plot
    cbar = plt.colorbar(m, orientation="horizontal", shrink=0.5)
    if varii == "perm":
        cbar.set_label("Log K(mD)", fontsize=12)
        ax.set_title(
            "Permeability Field with well locations", fontsize=12, weight="bold"
        )
    elif varii == "water Modulus":
        cbar.set_label("water saturation", fontsize=12)
        ax.set_title("water saturation -Modulus", fontsize=12, weight="bold")
    elif varii == "water Numerical":
        cbar.set_label("water saturation", fontsize=12)
        ax.set_title("water saturation - Numerical(Flow)", fontsize=12, weight="bold")
    elif varii == "water diff":
        cbar.set_label("unit", fontsize=12)
        ax.set_title(
            "water saturation - (Numerical(Flow) -Modulus))", fontsize=12, weight="bold"
        )

    elif varii == "oil Modulus":
        cbar.set_label("Oil saturation", fontsize=12)
        ax.set_title("Oil saturation -Modulus", fontsize=12, weight="bold")

    elif varii == "oil Numerical":
        cbar.set_label("Oil saturation", fontsize=12)
        ax.set_title("Oil saturation - Numerical(Flow)", fontsize=12, weight="bold")

    elif varii == "oil diff":
        cbar.set_label("unit", fontsize=12)
        ax.set_title(
            "oil saturation - (Numerical(Flow) -Modulus))", fontsize=12, weight="bold"
        )

    elif varii == "gas Modulus":
        cbar.set_label("Gas saturation", fontsize=12)
        ax.set_title("Gas saturation -Modulus", fontsize=12, weight="bold")

    elif varii == "gas Numerical":
        cbar.set_label("Gas saturation", fontsize=12)
        ax.set_title("Gas saturation - Numerical(Flow)", fontsize=12, weight="bold")

    elif varii == "gas diff":
        cbar.set_label("unit", fontsize=12)
        ax.set_title(
            "gas saturation - (Numerical(Flow) -Modulus))", fontsize=12, weight="bold"
        )

    elif varii == "pressure Modulus":
        cbar.set_label("pressure", fontsize=12)
        ax.set_title("Pressure -Modulus", fontsize=12, weight="bold")

    elif varii == "pressure Numerical":
        cbar.set_label("pressure", fontsize=12)
        ax.set_title("Pressure -Numerical(Flow)", fontsize=12, weight="bold")

    elif varii == "pressure diff":
        cbar.set_label("unit", fontsize=12)
        ax.set_title(
            "Pressure - (Numerical(Flow) -Modulus))", fontsize=12, weight="bold"
        )

    elif varii == "porosity":
        cbar.set_label("porosity", fontsize=12)
        ax.set_title("Porosity Field", fontsize=12, weight="bold")
    cbar.mappable.set_clim(minii, maxii)


# list of FNO dataset url ids on drive: https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-


def sort_key(s):
    """Extract the number from the filename for sorting."""
    return int(re.search(r"\d+", s).group())


def load_FNO_dataset(path, input_keys, output_keys, n_examples=None):
    "Loads a FNO dataset"

    if not path.endswith(".hdf5"):
        raise Exception(
            ".hdf5 file required: please use utilities.preprocess_FNO_mat to convert .mat file"
        )

    # load data
    path = to_absolute_path(path)
    data = h5py.File(path, "r")
    _ks = [k for k in data.keys() if not k.startswith("__")]
    print(f"loaded: {path}\navaliable keys: {_ks}")

    # parse data
    invar, outvar = dict(), dict()
    for d, keys in [(invar, input_keys), (outvar, output_keys)]:
        for k in keys:

            # get data
            x = data[k]  # N, C, H, W
            x = x.astype(np.float16)

            # cut examples out
            if n_examples is not None:
                x = x[:n_examples]

            # print out normalisation values
            print(f"selected key: {k}, mean: {x.mean():.5e}, std: {x.std():.5e}")

            d[k] = x
    del data

    return (invar, outvar)


def load_FNO_dataset2(
    path, input_keys, output_keys1, output_keys2, output_keys3, n_examples=None
):
    "Loads a FNO dataset"

    if not path.endswith(".hdf5"):
        raise Exception(
            ".hdf5 file required: please use utilities.preprocess_FNO_mat to convert .mat file"
        )

    # load data
    path = to_absolute_path(path)
    data = h5py.File(path, "r")
    _ks = [k for k in data.keys() if not k.startswith("__")]
    print(f"loaded: {path}\navaliable keys: {_ks}")

    # parse data
    invar, outvar1, outvar2, outvar3 = dict(), dict(), dict(), dict()
    for d, keys in [
        (invar, input_keys),
        (outvar1, output_keys1),
        (outvar2, output_keys2),
        (outvar3, output_keys3),
    ]:
        for k in keys:

            # get data
            x = data[k]  # N, C, H, W
            x = x.astype(np.float16)

            # cut examples out
            if n_examples is not None:
                x = x[:n_examples]

            # print out normalisation values
            print(f"selected key: {k}, mean: {x.mean():.5e}, std: {x.std():.5e}")

            d[k] = x
    del data

    return (invar, outvar1, outvar2, outvar3)


def load_FNO_dataset2d(path, input_keys, output_keys1, n_examples=None):
    "Loads a FNO dataset"

    if not path.endswith(".hdf5"):
        raise Exception(
            ".hdf5 file required: please use utilities.preprocess_FNO_mat to convert .mat file"
        )

    # load data
    path = to_absolute_path(path)
    data = h5py.File(path, "r")
    _ks = [k for k in data.keys() if not k.startswith("__")]
    print(f"loaded: {path}\navaliable keys: {_ks}")

    # parse data
    invar, outvar1 = dict(), dict()
    for d, keys in [(invar, input_keys), (outvar1, output_keys1)]:
        for k in keys:

            # get data
            x = data[k]  # N, C, H, W
            x = x.astype(np.float16)

            # cut examples out
            if n_examples is not None:
                x = x[:n_examples]

            # print out normalisation values
            print(f"selected key: {k}, mean: {x.mean():.5e}, std: {x.std():.5e}")

            d[k] = x
    del data

    return (invar, outvar1)


def load_FNO_dataset2a(path, input_keys, output_keys1, n_examples=None):
    "Loads a FNO dataset"

    if not path.endswith(".hdf5"):
        raise Exception(
            ".hdf5 file required: please use utilities.preprocess_FNO_mat to convert .mat file"
        )

    # load data
    path = to_absolute_path(path)
    data = h5py.File(path, "r")
    _ks = [k for k in data.keys() if not k.startswith("__")]
    print(f"loaded: {path}\navaliable keys: {_ks}")

    # parse data
    invar, outvar1 = dict(), dict()
    for d, keys in [(invar, input_keys), (outvar1, output_keys1)]:
        for k in keys:

            # get data
            x = data[k]  # N, C, H, W
            x = x.astype(np.float16)

            # cut examples out
            if n_examples is not None:
                x = x[:n_examples]

            # print out normalisation values
            print(f"selected key: {k}, mean: {x.mean():.5e}, std: {x.std():.5e}")

            d[k] = x
    del data

    return (invar, outvar1)


def _download_file_from_google_drive(id, path):
    "Downloads a file from google drive"

    # use gdown library to download file
    gdown.download(id=id, output=path)


def preprocess_FNO_mat2(path):
    "Convert a FNO .gz file to a hdf5 file, adding extra dimension to data arrays"

    assert path.endswith(".gz")
    # data = scipy.io.loadmat(path)
    with gzip.open(path, "rb") as f1:
        data = pickle.load(f1)

    ks = [k for k in data.keys() if not k.startswith("__")]
    with h5py.File(path[:-4] + ".hdf5", "w") as f:
        for k in ks:
            # x = np.expand_dims(data[k], axis=1)  # N, C, H, W
            x = data[k]
            f.create_dataset(
                k, data=x, dtype="float16", compression="gzip", compression_opts=9
            )  # note h5 files larger than .mat because no compression used


def preprocess_FNO_mat(path):
    "Convert a FNO .mat file to a hdf5 file, adding extra dimension to data arrays"

    assert path.endswith(".mat")
    data = scipy.io.loadmat(path)
    ks = [k for k in data.keys() if not k.startswith("__")]
    with h5py.File(path[:-4] + ".hdf5", "w") as f:
        for k in ks:
            # x = np.expand_dims(data[k], axis=1)  # N, C, H, W
            x = data[k]
            x = x.astype(np.float16)
            f.create_dataset(
                k, data=x, dtype="float16", compression="gzip", compression_opts=9
            )  # note h5 files larger than .mat because no compression used


def gaussianizeit(input1):
    numrows1 = len(input1)
    numcols = len(input1[0])
    newbig = np.zeros((numrows1, numcols))
    for i in range(numcols):
        input11 = input1[:, i]
        newX = norm.ppf(rankdata(input11) / (len(input11) + 1))
        newbig[:, i] = newX.T
    return newbig


def interpolatebetween(xtrain, cdftrain, xnew):
    numrows1 = len(xnew)
    numcols = len(xnew[0])
    norm_cdftest2 = np.zeros((numrows1, numcols))
    for i in range(numcols):
        f = interpolate.interp1d((xtrain[:, i]), cdftrain[:, i], kind="linear")
        cdftest = f(xnew[:, i])
        norm_cdftest2[:, i] = np.ravel(cdftest)
    return norm_cdftest2


def getoptimumkk(X, i, training_master, oldfolder):
    distortions = []
    Kss = range(1, 10)

    for k in Kss:
        kmeanModel = MiniBatchKMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(
            sum(np.min(cdist(X, kmeanModel.cluster_centers_, "euclidean"), axis=1))
            / X.shape[0]
        )

    myarray = np.array(distortions)

    knn = KneeLocator(
        Kss, myarray, curve="convex", direction="decreasing", interp_method="interp1d"
    )
    kuse = knn.knee

    # Plot the elbow
    plt.figure(figsize=(10, 10))
    plt.plot(Kss, distortions, "bx-")
    plt.xlabel("cluster size")
    plt.ylabel("Distortion")
    plt.title("Elbow Method showing the optimal n_clusters for machine %d" % (i))
    os.chdir(training_master)
    plt.savefig("machine_%d.jpg" % (i + 1))
    os.chdir(oldfolder)
    # plt.show()
    plt.close()
    plt.clf()
    return kuse


def getoptimumkcost(X, i, training_master, oldfolder):
    distortions = []
    Kss = range(1, 10)

    for k in Kss:
        kmeanModel = MiniBatchKMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(
            sum(np.min(cdist(X, kmeanModel.cluster_centers_, "euclidean"), axis=1))
            / X.shape[0]
        )

    myarray = np.array(distortions)

    knn = KneeLocator(
        Kss, myarray, curve="convex", direction="decreasing", interp_method="interp1d"
    )
    kuse = knn.knee

    # Plot the elbow
    plt.figure(figsize=(10, 10))
    plt.plot(Kss, distortions, "bx-")
    plt.xlabel("cluster size")
    plt.ylabel("Distortion")
    plt.title("Elbow Method showing the optimal n_clusters for machine %d" % (i))
    os.chdir(training_master)
    plt.savefig("machine_Energy__%d.jpg" % (i + 1))
    os.chdir(oldfolder)
    plt.show()
    return kuse


def best_fit(X, Y):

    xbar = sum(X) / len(X)
    ybar = sum(Y) / len(Y)
    n = len(X)  # or len(Y)
    numer = sum([xi * yi for xi, yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2
    b = numer / denum
    a = ybar - b * xbar

    print("best fit line:\ny = {:.2f} + {:.2f}x".format(a, b))
    return a, b


def Performance_plot_cost(CCR, Trued, stringg, training_master, oldfolder):

    CoDview = np.zeros((1, Trued.shape[1]))
    R2view = np.zeros((1, Trued.shape[1]))

    plt.figure(figsize=(40, 40))

    for jj in range(Trued.shape[1]):
        print(" Compute L2 and R2 for the machine _" + str(jj + 1))

        clementanswer2 = np.reshape(CCR[:, jj], (-1, 1))
        outputtest2 = np.reshape(Trued[:, jj], (-1, 1))
        numrowstest = len(outputtest2)
        outputtest2 = np.reshape(outputtest2, (-1, 1))
        Lerrorsparse = (
            LA.norm(outputtest2 - clementanswer2) / LA.norm(outputtest2)
        ) ** 0.5
        L_22 = 1 - (Lerrorsparse**2)
        # Coefficient of determination
        outputreq = np.zeros((numrowstest, 1))
        for i in range(numrowstest):
            outputreq[i, :] = outputtest2[i, :] - np.mean(outputtest2)
        CoDspa = 1 - (LA.norm(outputtest2 - clementanswer2) / LA.norm(outputreq))
        CoD2 = 1 - (1 - CoDspa) ** 2
        print("")

        CoDview[:, jj] = CoD2
        R2view[:, jj] = L_22

        jk = jj + 1
        plt.subplot(9, 9, jk)
        # palette = copy(plt.get_cmap('inferno_r'))
        # palette.set_under('white')  # 1.0 represents not transparent
        # palette.set_over('black')  # 1.0 represents not transparent
        vmin = min(np.ravel(outputtest2))
        vmax = max(np.ravel(outputtest2))
        sc = plt.scatter(
            np.ravel(clementanswer2),
            np.ravel(outputtest2),
            c=np.ravel(outputtest2),
            vmin=vmin,
            vmax=vmax,
            s=35,
            cmap="inferno_r",
        )
        plt.colorbar(sc)
        plt.title("Energy_" + str(jj), fontsize=9)
        plt.ylabel("Machine", fontsize=9)
        plt.xlabel("True data", fontsize=9)
        a, b = best_fit(
            np.ravel(clementanswer2),
            np.ravel(outputtest2),
        )
        yfit = [a + b * xi for xi in np.ravel(clementanswer2)]
        plt.plot(np.ravel(clementanswer2), yfit, color="r")
        plt.annotate(
            "R2= %.3f" % CoD2,
            (0.8, 0.2),
            xycoords="axes fraction",
            ha="center",
            va="center",
            size=9,
        )

    CoDoverall = (np.sum(CoDview, axis=1)) / Trued.shape[1]
    R2overall = (np.sum(R2view, axis=1)) / Trued.shape[1]
    os.chdir(training_master)
    plt.savefig("%s.jpg" % stringg)
    os.chdir(oldfolder)
    return CoDoverall, R2overall, CoDview, R2view


def run_model(model, inn, ouut, i, training_master, oldfolder):
    model.fit(inn, ouut)
    filename = "Classifier_%d.bin" % i
    os.chdir(training_master)
    model.save_model(filename)
    os.chdir(oldfolder)
    return model


def run_modelcost(model, inn, ouut, i, training_master, oldfolder):
    model.fit(inn, ouut)
    filename = "Classifiercost_%d.bin" % i
    os.chdir(training_master)
    model.save_model(filename)
    os.chdir(oldfolder)
    return model


def startit(i, outpuut2, inpuut2, training_master, oldfolder, degg):
    print("")
    print("Starting CCR training machine %d" % (i + 1))
    useeo = outpuut2[:, i]
    useeo = np.reshape(useeo, (-1, 1), "F")

    usein = inpuut2
    usein = np.reshape(usein, (-1, 90), "F")  # 9+4

    clust = CCR_Machine(usein, useeo, i, training_master, oldfolder, degg)

    bigs = clust
    return bigs
    print("")
    print("Finished training machine %d" % (i + 1))


def endit(i, testt, training_master, oldfolder, pred_type, degg, big):
    print("")
    print("Starting prediction from machine %d" % (i + 1))

    numcols = len(testt[0])
    clemzz = PREDICTION_CCR__MACHINE(
        i, big, testt, numcols, training_master, oldfolder, pred_type, degg
    )

    print("")
    print("Finished Prediction from machine %d" % (i + 1))
    return clemzz


def fit_machine(a0, b0, deg):

    # dim = a0.shape[1]
    poly_features = PolynomialFeatures(degree=deg, include_bias=False)
    x_poly = poly_features.fit_transform(a0)

    # Fit a linear regression model
    lin_reg = LinearRegression()
    lin_reg.fit(x_poly, b0)
    return lin_reg, poly_features


def predict_machine(a0, deg, model, model2):
    x_new_poly = model2.transform(a0)
    y_new = model.predict(x_new_poly)
    return y_new


def CCR_Machine(inpuutj, outputtj, ii, training_master, oldfolder, degg):
    # print('Starting CCR')
    model = xgb.XGBClassifier(n_estimators=1000)
    # import numpy as np
    # import pickle
    X = inpuutj
    y = outputtj
    numruth = len(X[0])

    y_traind = y
    scaler1a = MinMaxScaler(feature_range=(0, 1))
    (scaler1a.fit(X))
    X = scaler1a.transform(X)
    scaler2a = MinMaxScaler(feature_range=(0, 1))
    (scaler2a.fit(y))
    y = scaler2a.transform(y)
    yruth = y
    os.chdir(training_master)
    filenamex = "clfx_%d.asv" % ii
    filenamey = "clfy_%d.asv" % ii
    pickle.dump(scaler1a, open(filenamex, "wb"))
    pickle.dump(scaler2a, open(filenamey, "wb"))
    os.chdir(oldfolder)
    y_traind = numruth * 10 * y
    matrix = np.concatenate((X, y_traind), axis=1)
    k = getoptimumkk(matrix, ii, training_master, oldfolder)
    nclusters = k
    # nclusters=3
    print("Optimal k is: ", nclusters)
    kmeans = MiniBatchKMeans(n_clusters=nclusters, max_iter=2000).fit(matrix)
    filename = "Clustering_%d.asv" % ii
    os.chdir(training_master)
    pickle.dump(kmeans, open(filename, "wb"))
    os.chdir(oldfolder)
    dd = kmeans.labels_
    dd = dd.T
    dd = np.reshape(dd, (-1, 1))
    # -------------------#---------------------------------#
    inputtrainclass = X
    outputtrainclass = np.reshape(dd, (-1, 1))
    run_model(model, inputtrainclass, outputtrainclass, ii, training_master, oldfolder)
    # print('Split for classifier problem')
    X_train = X
    y_train = dd
    # -------------------Regression----------------#
    # print('Learn regression of the clusters with different labels from k-means ' )

    for i in range(nclusters):
        print("-- Learning cluster: " + str(i + 1) + " | " + str(nclusters))
        label0 = (np.asarray(np.where(y_train == i))).T
        # model0=xgb.XGBRegressor(n_estimators=2000)
        # model0=np.empty([1,2],dtype=object)

        a0 = X_train[label0[:, 0], :]
        a0 = np.reshape(a0, (-1, numruth), "F")
        b0 = yruth[label0[:, 0], :]
        b0 = np.reshape(b0, (-1, 1), "F")
        if a0.shape[0] != 0 and b0.shape[0] != 0:
            # model0.fit(a0, b0,verbose=False)
            theta, con1 = fit_machine(a0, b0, degg)

        filename = "Regressor_Machine_" + str(ii) + "_Cluster_" + str(i) + ".pkl"
        filename2 = "Regressor_Features_" + str(ii) + "_Cluster_" + str(i) + ".pkl"
        os.chdir(training_master)
        # sio.savemat(filename, {'model0':model0})

        with open(filename, "wb") as model_file:
            pickle.dump(theta, model_file)

        # Save the transformer
        with open(filename2, "wb") as transformer_file:
            pickle.dump(con1, transformer_file)

        os.chdir(oldfolder)
    return nclusters
    # print('Finished CCR')


def PREDICTION_CCR__MACHINE(
    ii, nclusters, inputtest, numcols, training_master, oldfolder, pred_type, deg
):
    # import numpy as np
    # ii=0
    # nclusters=2
    # inputtest=X_test2
    # print('Starting Prediction')
    # print('-- Column : ' + str(ii+1) + ' | ' + str(66))
    filename1 = "Classifier_%d.bin" % ii
    filenamex = "clfx_%d.asv" % ii
    filenamey = "clfy_%d.asv" % ii
    os.chdir(training_master)
    loaded_model = xgb.Booster({"nthread": 4})  # init model
    clfx = pickle.load(open(filenamex, "rb"))
    clfy = pickle.load(open(filenamey, "rb"))
    loaded_model.load_model(filename1)  # load data
    os.chdir(oldfolder)
    inputtest = clfx.transform(inputtest)
    labelDA = loaded_model.predict(xgb.DMatrix(inputtest))
    if nclusters == 2:
        labelDAX = 1 - labelDA
        labelDA = np.reshape(labelDA, (-1, 1))
        labelDAX = np.reshape(labelDAX, (-1, 1))
        labelDA = np.concatenate((labelDAX, labelDA), axis=1)

    numrowstest = len(inputtest)
    clementanswer = np.zeros((numrowstest, 1))
    # numcols=13
    if pred_type == 1:  # Hard prediction
        labelDA = np.argmax(labelDA, axis=-1)
        labelDA = np.reshape(labelDA, (-1, 1), "F")
        for i in range(nclusters):
            # print('-- Predicting cluster: ' + str(i+1) + ' | ' + str(nclusters))
            filename2 = "Regressor_Machine_" + str(ii) + "_Cluster_" + str(i) + ".pkl"
            filename2a = "Regressor_Features_" + str(ii) + "_Cluster_" + str(i) + ".pkl"
            os.chdir(training_master)
            # Load the model
            with open(filename2, "rb") as model_file:
                model0 = pickle.load(model_file)

            # Load the transformer
            with open(filename2a, "rb") as transformer_file:
                modell0 = pickle.load(transformer_file)

            os.chdir(oldfolder)
            labelDA0 = (np.asarray(np.where(labelDA == i))).T
            #    ##----------------------##------------------------##
            a00 = inputtest[labelDA0[:, 0], :]
            a00 = np.reshape(a00, (-1, numcols), "F")
            if a00.shape[0] != 0:
                clementanswer[labelDA0[:, 0], :] = np.reshape(
                    predict_machine(a00, deg, model0, modell0), (-1, 1)
                )

        clementanswer = clfy.inverse_transform(clementanswer)
    else:  # soft prediction
        # deg=4
        big_out = np.zeros((numrowstest, nclusters))
        for i in range(nclusters):
            # print('-- predicting cluster: ' + str(i+1) + ' | ' + str(nclusters))
            filename2 = "Regressor_Machine_" + str(ii) + "_Cluster_" + str(i) + ".pkl"
            filename2a = "Regressor_Features_" + str(ii) + "_Cluster_" + str(i) + ".pkl"
            os.chdir(training_master)
            with open(filename2, "rb") as model_file:
                model0 = pickle.load(model_file)

            # Load the transformer
            with open(filename2a, "rb") as transformer_file:
                modell0 = pickle.load(transformer_file)
            os.chdir(oldfolder)
            aa = np.reshape(predict_machine(inputtest, deg, model0, modell0), (-1, 1))
            aanew = np.multiply(aa, np.reshape(labelDA[:, i], (-1, 1)))
            # aanew[aanew<=0] = 0
            big_out[:, i] = np.ravel(aanew)
        clementanswer = np.reshape(np.sum(big_out, axis=1), (-1, 1), "F")
        clementanswer[clementanswer <= 0] = 0
        # clementanswer=clfy.inverse_transform(clementanswer)
    return clementanswer
    # print('Finished prediction')
