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


def load_testing_data(path: str, num_pred: int):
    """
    Load testing data to perform inference on pre-trained network.

    Parameters
    ----------
    path : str
        Path to folder containing testing data (input.py and ground_truth.py)
    num_pred : int
        Number of samples to load to perform inference on.

    Returns
    -------
    Two numpy.ndarrays containing aliased testing samples and ground truth samples.
    """
    x_path = os.path.join(path, 'input.npy')
    y_path = os.path.join(path, 'ground_truth.npy')
    x_test, y_test = np.load(x_path), np.load(y_path)
    num_samples = x_test.shape[0]
    rand_ind = np.random.randint(low=0, high=num_samples, size=num_pred)

    return np.load(x_path)[rand_ind], np.load(y_path)[rand_ind]


def compat_check(model: Model, x_test: np.ndarray):
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


def freq_correct(x_test: np.ndarray, y_pred: np.ndarray, skip_factor: int):
    """
    Perform low-frequency k-space correction on predicted samples as described in 'Deep learning for undersampled MRI
    reconstruction' by Hyun et. al.

    Parameters
    ----------
    x_test : numpy.ndarray
        ndarray containing aliased test samples.
    y_pred : numpy.ndarray
        ndarray containing model's de-aliased predictions.
    skip_factor : int
        Undersampling factor that the model was pre-trained on.

    Returns
    -------
    y_predict : numpy.ndarray
        ndarray of frequency-corrected samples of y_pred
    """
    size = y_pred.shape[1]
    num_lines = skip_factor / 100 * size
    num_lines = int(np.power(2, np.ceil(np.log2(num_lines))))
    start = int((size - num_lines) / 2)
    end = int((size + num_lines) / 2)
    # print('{}% of {}; {} lines from {} to {}'.format(skip_factor, size, num_lines, start, end))

    ft_x_test = np.fft.fftshift(np.fft.fft2(np.squeeze(x_test), axes=(1, 2)), axes=(1, 2))
    ft_y_predict = np.fft.fftshift(np.fft.fft2(np.squeeze(y_pred), axes=(1, 2)), axes=(1, 2))
    ft_y_predict[:, 0:start:skip_factor, :] = ft_x_test[:, 0:start:skip_factor, :]
    ft_y_predict[:, start:end, :] = ft_x_test[:, start:end, :]
    ft_y_predict[:, end::skip_factor, :] = ft_x_test[:, end::skip_factor, :]

    y_predict = np.fft.ifft2(np.fft.ifftshift(ft_y_predict, axes=(1, 2)), axes=(1, 2))
    y_predict = np.abs(y_predict)
    y_predict = np.expand_dims(y_predict, axis=-1)
    assert len(y_predict.shape) == len(y_pred.shape)

    return y_predict


def plot(x_test: np.ndarray, y_pred: np.ndarray, y_test: np.ndarray, num_pred: int):
    """
    Plot `num_pred` number of aliased samples, corresponding predictions and ground truth and in Matplotlib.

    Parameters
    ----------
    x_test : numpy.ndarray
        ndarray containing aliased test samples.
    y_pred : numpy.ndarray
        ndarray of mode's de-aliased predictions.
    y_test : numpy.ndarray
        ndarray containing ground truth samples.
    num_pred : int
        Number of predictions to perform inference on.
    """
    grid_size = int('{}30'.format(num_pred))

    plt.figure(figsize=(8, 8))

    for x in range(num_pred):
        grid_size += 1
        plt.subplot(grid_size)
        plt.imshow(np.squeeze(x_test[x]), cmap='gray')
        if x == 0:
            plt.title('Aliased input')

        grid_size += 1
        plt.subplot(grid_size)
        plt.imshow(np.squeeze(y_test[x]), cmap='gray')
        if x == 0:
            plt.title('Ground truth')

        grid_size += 1
        plt.subplot(grid_size)
        plt.imshow(np.squeeze(y_pred[x]), cmap='gray')
        if x == 0:
            plt.title('Reconstruction')

    plt.show()


def inference(model_path: str, testing_data_path: str, skip_factor: int, num_pred: int):
    """
    Perform inference on pre-trained network, compute and display time to perform inference and plot results.

    Parameters
    ----------
    model_path : keras.models.Model
        Path to load pre-trained Keras model.
    testing_data_path : str
        Path to folder containing input.npy and ground_truth.npy.
    skip_factor : int
        Undersampling factor of the dataset that the model was pre-trained on.
    num_pred : int
        Number of test samples to perform inference on.
    """
    model = load_model(model_path)  # Load model
    x_test, y_test = load_testing_data(testing_data_path, num_pred)  # Load testing data
    compat_check(model, x_test)  # Check if expected shape of input and actual shape of testing input  match

    start = time.time()
    y_pred = model.predict(x_test)
    end = time.time()
    diff = end - start
    y_pred = freq_correct(x_test, y_pred, skip_factor)

    print('Inference took {:.3g}s'.format(diff))
    plot(x_test, y_pred, y_test, num_pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DRUNCK: Deep-learning Reconstruction of UNdersampled Cartesian K-space data')
    parser.add_argument('model_path', type=str, help='Path to Keras model saved as .hdf5')
    parser.add_argument('test_data_path', type=str, help='Path to folder containing input.npy and ground_truth.npy')
    parser.add_argument('-skip', '--skip_factor', type=int, default=4, help='Undersampling factor')
    parser.add_argument('-npred', '--num_predictions', type=int, default=3, help='Number of samples to predict')
    args = parser.parse_args()

    model_path = args.model_path
    test_data_path = args.test_data_path
    skip_factor = args.skip_factor
    num_pred = args.num_predictions

    inference(model_path, test_data_path, skip_factor, num_pred)
