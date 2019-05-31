"""
Convert NIFTI data to `numpy.ndarray` and undersample data.

Author: Keerthi Sravan Ravi
Date: 03/22/2019
Version 0.1
Copyright of the Board of Trustees of  Columbia University in the City of New York.
"""
import argparse
import os

import PIL
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def load_dataset_from_nifti(nifti_path: str, img_size: int = 128) -> np.ndarray:
    """
    Make dataset by reading NIFTI files from `nifti_path` and resizing each image to `img_size`x`img_size`.

    Parameters
    ----------
    nifti_path : str
        Path to folder containing NIFTI files.
    img_size : int
        Desired size of images in dataset. Images read from NIFTI files will be resized.

    Returns
    -------
    dataset : numpy.ndarray
        ndarray of images converted from NIFTI files.
    """
    print('Looking in {}... '.format(nifti_path), end='')
    filenames = os.listdir(nifti_path)
    num_files = len(filenames)
    print('Found {} files.'.format(num_files))

    slice_min, slice_max = 64, 200  # Slices of interest from volume data
    dataset = np.empty(shape=(0, img_size, img_size, 1))  # Empty array

    counter = 0
    for file in filenames:
        if not file.endswith('DS_Store'):
            counter = counter + 1
            n = nib.load(os.path.join(nifti_path, file))  # Load NIFTI
            vol = n.get_data()[:, :, slice_min:slice_max + 1]  # Get image
            print('File {}/{}, shape: {}'.format(counter, num_files, vol.shape))

            for i in range(vol.shape[2]):
                img = vol[:, :, i]
                img = np.array(PIL.Image.fromarray(img).resize((img_size, img_size)))
                img = img[np.newaxis, :, :, np.newaxis]
                dataset = np.append(dataset, img, axis=0)
    print(dataset.shape, dataset.dtype)
    print('\n')
    return dataset


def undersample(dataset: np.ndarray, low_freq_pc: float, reduction_factor: int) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Undersample `dataset` as follows:
    1. Obtain `ft_dataset` - Fourier Transform of `dataset`
    2. Obtain `ft_dataset_undersampled` - undersample of `ft_dataset`
    3. Add `low_freq_pc` low-frequency k-space components from `ft_dataset` to `ft_dataset_undersampled`.
    4. Obtain `dataset_undersampled` - Inverse Fourier Transform of `ft_dataset_undersampled`.

    Parameters
    ----------
    dataset : numpy.ndarray
        ndarray of dataset samples.
    low_freq_pc : float
        Percentage of low-frequency fully-sampled k-space values to add to undersampled k-space.
    reduction_factor : int
        Undersampling factor.

    Returns
    -------
    ft_dataset : numpy.ndarray
        ndarray of Fourier Transform of `dataset`.
    dataset_undersampled : numpy.ndarray
        ndarray of undersampled `dataset`.
    ft_dataset_undersampled : numpy.ndarray
        Fourier Transform of `dataset_undersampled`.
    """
    # Fourier transform
    print('FT... ', end='')
    ft_dataset = np.fft.fftshift(np.fft.fft2(dataset, axes=(1, 2)), axes=(1, 2))
    print(ft_dataset.shape, ft_dataset.dtype, end='')
    print('. Done.')

    # Undersample the Fourier Transform and add low-frequencies
    num_rows = dataset.shape[1]
    num_lines = low_freq_pc * num_rows
    num_lines = int(np.power(2, np.ceil(np.log2(num_lines))))
    start = int((num_rows - num_lines) / 2)
    end = int((num_rows + num_lines) / 2)
    print('{}% of {}; {} lines from {} to {}'.format(reduction_factor, num_rows, num_lines, start, end))

    print('Undersampling... ', end='')
    ft_dataset_undersampled = np.zeros_like(ft_dataset)
    ft_dataset_undersampled[:, 0:start:reduction_factor, :] = ft_dataset[:, 0:start:reduction_factor, :]
    ft_dataset_undersampled[:, start:end, :] = ft_dataset[:, start:end, :]
    ft_dataset_undersampled[:, end::reduction_factor, :] = ft_dataset[:, end::reduction_factor, :]
    print(ft_dataset_undersampled.shape, ft_dataset_undersampled.dtype, end='')
    print('. Done.')

    # Aliased reconstruction
    print('Inverse FT... ', end='')
    dataset_undersampled = np.fft.ifft2(np.fft.ifftshift(ft_dataset_undersampled, axes=(1, 2)), axes=(1, 2))
    print(dataset_undersampled.shape, dataset_undersampled.dtype, end='')
    print(' Done.\n')

    return ft_dataset, dataset_undersampled, ft_dataset_undersampled


def normalise_dataset(dataset: np.ndarray) -> np.ndarray:
    """
    Normalised `dataset` to [0, 1].

    Parameters
    ----------
    dataset : numpy.ndarray
        ndarray of dataset samples.

    Returns
    -------
    normalised_dataset : numpy.ndarray
        ndarray of `dataset` normalised to [0, 1].
    """
    print('Normalising... ', end='')
    m1 = np.min(dataset.min(axis=1), axis=1)
    m1 = m1[:, np.newaxis, np.newaxis]
    m2 = np.max(dataset.max(axis=1), axis=1)
    m2 = m2[:, np.newaxis, np.newaxis]
    normalised_dataset = dataset - m1
    normalised_dataset *= 255 / (m2 - m1)
    print(normalised_dataset.shape, normalised_dataset.dtype, end='')
    print(' Done.\n')

    return normalised_dataset


def plot(dataset: np.ndarray, ft_dataset: np.ndarray, dataset_undersampled: np.ndarray,
         ft_dataset_undersampled: np.ndarray, img_ind: int):
    """
    Plot fully-sampled and under-sampled data sample at `img_ind` position with corresponding k-space data.

    Parameters
    ----------
    dataset : numpy.ndarray
        ndarray of dataset samples.
    ft_dataset : numpy.ndarray
        ndarray of Fourier Transform of `dataset`.
    dataset_undersampled : numpy.ndarray
        ndarray of undersampled `dataset`.
    ft_dataset_undersampled : numpy.ndarray
        Fourier Transform of `dataset_undersampled`.
    img_ind : int
        Index of data sample to plot.
    """
    plt.figure(figsize=(10, 8))
    plt.subplot(221)
    plt.title('Fully-sampled')
    plt.imshow(dataset[img_ind, :, :, 0], cmap='gray')
    plt.subplot(222)
    plt.title('Under-sampled')
    plt.imshow(np.abs(dataset_undersampled[img_ind, :, :, 0]), cmap='gray')
    plt.colorbar()
    plt.subplot(223)
    plt.imshow(np.abs(ft_dataset[img_ind, :, :, 0]), cmap='gray', vmax=5000)
    plt.subplot(224)
    plt.imshow(np.abs(ft_dataset_undersampled[img_ind, :, :, 0]), cmap='gray', vmax=5000)
    plt.show()

    plt.show()


def save2disk(filename1: str, file1: np.ndarray, filename2: str, file2: np.ndarray, save_path: str):
    """
    Save `file1` and `file2` to disk as `filename1` and `filename2 at `save_path`.

    Parameters
    ----------
    filename1 : str
        Filename of `file1`.
    file1 : numpy.ndarray
        ndarray to be saved to disk.
    filename2 : str
        Filename of `file2`.
    file2 : numpy.ndarray
        ndarray to be saved to disk.
    save_path : str
        Path to save ndarray files to.
    """
    print('Saving to {}...'.format(os.path.join(save_path)), end='')
    path = os.path.join(save_path, filename1)
    np.save(path, file1)
    path = os.path.join(save_path, filename2)
    np.save(path, file2)
    print(' Done.\n')


def main(nifti_path: str, img_size: int, low_freq_pc: float, save_path: str, reduction_factor: int,
         plot_flag: bool = True):
    """
    1. Load NIFTI data as `numpy.ndarray` and resize each image to `img_size`x`img_size`.
    2. Undersample by `skip_factor` and add `low_freq_pc` low-frequency k-space values.
    3. Save files to disk.

    Parameters
    ----------
    nifti_path : str
        Path to folder containing NIFTI files.
    img_size : int
        Desired size of images in dataset. Images read from NIFTI files will be resized.
    low_freq_pc : float
        Percentage of low-frequency fully-sampled k-space values to add to undersampled k-space.
    save_path : str
        Path to save ndarray files.
    reduction_factor : int
        Undersampling factor.
    plot_flag : bool
        Boolean flag to plot `img_ind` sample of dataset, undersampled dataset and corresponding k-space.
    """
    if save_path is not str() and not os.path.exists(save_path):
        os.makedirs(os.path.abspath(save_path))

    dataset = load_dataset_from_nifti(nifti_path=nifti_path, img_size=img_size)
    dataset = normalise_dataset(dataset=dataset)
    result = undersample(dataset=dataset, low_freq_pc=low_freq_pc, reduction_factor=reduction_factor)
    ft_dataset, dataset_undersampled, ft_dataset_undersampled = result
    dataset_undersampled = np.abs(dataset_undersampled)
    dataset_undersampled = normalise_dataset(dataset=dataset_undersampled)

    if save_path is not str():
        save2disk(filename1='x.npy', file1=dataset_undersampled, filename2='y', file2=dataset, save_path=save_path)
    if plot_flag:
        plot(dataset=dataset, ft_dataset=ft_dataset, dataset_undersampled=dataset_undersampled,
             ft_dataset_undersampled=ft_dataset_undersampled, img_ind=123)

    return dataset_undersampled, dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DRUNCK: Deep-learning Reconstruction of UNdersampled Cartesian K-space data')
    parser.add_argument('nifti_path', type=str, help='Path to folder containing NIFTI files')
    parser.add_argument('img_size', type=int, help='Desired image size')
    parser.add_argument('low_frequency', type=float, help='Percentage of low-frequency k-space to add')
    parser.add_argument('-s', '--save_path', type=str, default=str(), help='Path to save converted .npy files')
    parser.add_argument('-r', '--reduction_factor', type=int, default=4, help='Undersampling factor')
    parser.add_argument('-p', '--plot', type=bool, default=True,
                        help='Plot one random dataset, undersampled dataset and corresponding k-space sample')
    args = parser.parse_args()

    nifti_path = args.nifti_path
    img_size = args.img_size
    low_freq_pc = args.low_frequency
    save_path = args.save_path
    reduction_factor = args.reduction_factor
    plot_flag = args.plot

    main(nifti_path, img_size, low_freq_pc, save_path, reduction_factor, plot_flag)
