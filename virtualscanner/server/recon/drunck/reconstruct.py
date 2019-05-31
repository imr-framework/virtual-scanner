"""
Perform inference on pre-trained Keras model.

Author: Keerthi Sravan Ravi
Date: 03/22/2019
Version 0.1
Copyright of the Board of Trustees of  Columbia University in the City of New York.
"""
from pathlib import Path
from time import time

import numpy as np
from PIL import Image
from keras.models import load_model


def __undersample(input_image):
    low_freq_pc = 0.04
    reduction_factor = 4
    size = input_image.shape[1]
    num_lines = low_freq_pc * size
    num_lines = int(np.power(2, np.ceil(np.log2(num_lines))))
    start = int((size - num_lines) / 2)
    end = int((size + num_lines) / 2)

    input_kspace = np.fft.fftshift(np.fft.fft2(input_image))
    aliased_kspace = np.zeros_like(input_kspace)
    aliased_kspace[0:start:reduction_factor] = input_kspace[0:start:reduction_factor]
    aliased_kspace[start:end] = input_kspace[start:end]
    aliased_kspace[end::reduction_factor] = input_kspace[end::reduction_factor]

    aliased_image = np.fft.ifft2(np.fft.ifftshift(aliased_kspace))
    aliased_image = np.abs(aliased_image)

    return aliased_image, aliased_kspace


def __freq_correct(x: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
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

    low_freq_pc = 0.04
    reduction_factor = 4
    size = len(y_pred)
    num_lines = low_freq_pc * size
    num_lines = int(np.power(2, np.ceil(np.log2(num_lines))))
    start = int((size - num_lines) / 2)
    end = int((size + num_lines) / 2)

    if x.dtype == np.complex128:  # Received kspace
        x_kspace = x
    else:  # Received image space
        x_kspace = np.fft.fftshift(np.fft.fft2(x))

    y_pred_kspace = np.fft.fftshift(np.fft.fft2(y_pred))
    y_pred_kspace[0:start:reduction_factor] = x_kspace[0:start:reduction_factor]
    y_pred_kspace[start:end] = x_kspace[start:end]
    y_pred_kspace[end::reduction_factor] = x_kspace[end::reduction_factor]

    y_pred = np.fft.ifft2(np.fft.ifftshift(y_pred_kspace))
    y_pred = np.abs(y_pred)

    return y_pred


def main(img_path: str, img_type: str) -> tuple:
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
    output_image : numpy.ndarray
        ndarray containing `num_pred` number of reconstructed samples.
    """

    img_path = Path(img_path)
    if img_path.exists():
        input_image = Image.open(str(img_path))
        if input_image.size != (256, 256):
            input_image = input_image.resize((256, 256))
        input_image = np.asarray(input_image)[..., 0]

        if img_type == 'GT':
            aliased_image, aliased_kspace = __undersample(input_image)
        elif img_type == 'US':
            aliased_image = input_image
            aliased_kspace = np.fft.fftshift(np.fft.fft2(aliased_image))
        else:
            raise ValueError('Unknown image type')

        # Load pre-trained Hyun model and perform inference
        model = load_model('./src/server/recon/drunck/assets/model.hdf5')
        output_image = model.predict(aliased_image[np.newaxis, ..., np.newaxis])
        output_image = np.squeeze(output_image)
        output_image = __freq_correct(aliased_kspace, output_image)

        t = time()

        # Save aliased input image
        aliased_filename = f'aliased_{t}.jpg'
        aliased_image = Image.fromarray(aliased_image).convert('RGB')
        aliased_image.save(f'./src/coms/coms_ui/static/recon/outputs/{aliased_filename}')

        # Save output image
        output_filename = f'output_{t}.jpg'
        output_image = Image.fromarray(output_image).convert('RGB')
        output_image.save(f'./src/coms/coms_ui/static/recon/outputs/{output_filename}')

        if img_type == 'GT':  # Save ground truth
            gt_filename = f'gt_{t}.jpg'
            input_image = Image.fromarray(input_image).convert('RGB')
            input_image.save(f'./src/coms/coms_ui/static/recon/outputs/{gt_filename}')
            return gt_filename, aliased_filename, output_filename

        return aliased_filename, output_filename
    else:
        raise ValueError('File not found')


# main('/Users/sravan953/Desktop/test.jpg', 'GT')
