"""
Unit test for DRUNCK.

Author: Keerthi Sravan Ravi
Date: 03/22/2019
Version 0.1
Copyright of the Board of Trustees of  Columbia University in the City of New York.
"""
import unittest

import numpy as np

import virtualscanner.server.recon.drunck.nifti2npy as nifti2npy


class DrunckTest(unittest.TestCase):
    def test_nifti2npy(self):
        nifti_path = '../../data/nifti/test/'
        img_size = 128
        low_freq_pc = 0.04
        reduction_factor = 4
        x1, y1 = nifti2npy.main(nifti_path=nifti_path, img_size=img_size, low_freq_pc=low_freq_pc, save_path=str(),
                                reduction_factor=reduction_factor, plot_flag=False)
        x2, y2 = np.load('../assets/unittest_nifti2npy_x.npy'), np.load('../assets/unittest_nifti2npy_y.npy')

        self.assertEqual(np.any(x1 - x2), 0)
        self.assertEqual(np.any(y1 - y2), 0)


if __name__ == '__main__':
    unittest.main()
