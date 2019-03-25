"""
Perform inference on pre-trained Keras model.

Author: Keerthi Sravan Ravi
Date: 03/22/2019
Version 0.1
Copyright of the Board of Trustees of  Columbia University in the City of New York.
"""
import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model
from keras.models import load_model


def __load_test_data(path: str, num_pred: int) -> (np.ndarray, np.ndarray):
    """
    Load testing data to perform inference on pre-trained network.

    Parameters
    ----------
    path : str
        Path to folder containing test data (input.py and ground_truth.py)
    num_pred : int
        Number of samples to load to perform inference on.

    Returns
    -------
    Two numpy.ndarrays containing aliased test samples and ground truth samples.
    """
    x_path = os.path.join(path, 'x.npy')
    y_path = os.path.join(path, 'y.npy')
    x_test, y_test = np.load(x_path), np.load(y_path)
    num_samples = x_test.shape[0]
    rand_ind = np.random.randint(low=0, high=num_samples, size=num_pred)

    return np.load(x_path)[rand_ind], np.load(y_path)[rand_ind]


def __compat_check(model: Model, x_test: np.ndarray):
    """
    Check if shape of samples in `x_test` match `model`'s samples.

    Parameters
    ----------
    model : keras.models.Model
        Pre-trained Keras model to perform inference on.
    x_test : numpy.ndarray
        ndarray containing aliased test samples.
    """
    img_shape = x_test.shape[1:]
    exp_shape = model.layers[0].input_shape[1:]
    assert img_shape == exp_shape


def __freq_correct(x_test: np.ndarray, y_pred: np.ndarray, reduction_factor: int) -> np.ndarray:
    """
    Perform low-frequency k-space correction on predicted samples as described in 'Deep learning for undersampled MRI
    reconstruction' by Hyun et. al.

    Parameters
    ----------
    x_test : numpy.ndarray
        ndarray containing aliased test samples.
    y_pred : numpy.ndarray
        ndarray containing model's de-aliased predictions.
    reduction_factor : int
        Undersampling factor that the model was pre-trained on.

    Returns
    -------
    y_pred : numpy.ndarray
        ndarray of frequency-corrected samples of y_pred
    """
    size = y_pred.shape[1]
    num_lines = reduction_factor / 100 * size
    num_lines = int(np.power(2, np.ceil(np.log2(num_lines))))
    start = int((size - num_lines) / 2)
    end = int((size + num_lines) / 2)
    # print('{}% of {}; {} lines from {} to {}'.format(skip_factor, size, num_lines, start, end))

    ft_x_test = np.fft.fftshift(np.fft.fft2(x_test, axes=(1, 2)), axes=(1, 2))
    ft_y_predict = np.fft.fftshift(np.fft.fft2(y_pred, axes=(1, 2)), axes=(1, 2))
    ft_y_predict[:, 0:start:reduction_factor, :] = ft_x_test[:, 0:start:reduction_factor, :]
    ft_y_predict[:, start:end, :] = ft_x_test[:, start:end, :]
    ft_y_predict[:, end::reduction_factor, :] = ft_x_test[:, end::reduction_factor, :]

    y_pred = np.fft.ifft2(np.fft.ifftshift(ft_y_predict, axes=(1, 2)), axes=(1, 2))
    y_pred = np.abs(y_pred)

    return y_pred


def main(model_path: str, test_data_path: str, reduction_factor: int, num_pred: int) -> np.ndarray:
    """
    Perform inference on pre-trained network, compute and display time to perform inference and plot results.

    Parameters
    ----------
    model_path : keras.models.Model
        Path to load pre-trained Keras model.
    test_data_path : str
        Path to folder containing input.npy and ground_truth.npy.
    reduction_factor : int
        Undersampling factor of the dataset that the model was pre-trained on.
    num_pred : int
        Number of test samples to perform inference on.

    Returns
    -------
    y_pred : numpy.ndarray
        ndarray containing `num_pred` number of reconstructed samples.
    """
    if num_pred <= 0:
        raise Exception('ValueError: num_pred should be at least1. You passed: {}'.format(num_pred))

    model = load_model(model_path)  # Load model
    x_test, y_test = __load_test_data(test_data_path, num_pred)  # Load test data
    __compat_check(model, x_test)  # Check if expected shape of input and actual shape of test input  match

    start = time.time()
    y_pred = model.predict(x_test)
    end = time.time()
    diff = end - start
    y_pred = __freq_correct(x_test, y_pred, reduction_factor)

    print('Inference took {:.3g}s'.format(diff))

    return y_pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DRUNCK: Deep-learning Reconstruction of UNdersampled Cartesian K-space data')
    parser.add_argument('model_path', type=str, help='Path to Keras model saved as .hdf5')
    parser.add_argument('test_data_path', type=str, help='Path to folder containing input.npy and ground_truth.npy')
    parser.add_argument('-r', '--reduction_factor', type=int, default=4, help='Undersampling factor')
    parser.add_argument('-npred', '--num_predictions', type=int, default=3, help='Number of samples to predict')
    args = parser.parse_args()

    model_path = args.model_path
    test_data_path = args.test_data_path
    reduction_factor = args.reduction_factor
    num_pred = args.num_predictions

    main(model_path, test_data_path, reduction_factor, num_pred)
