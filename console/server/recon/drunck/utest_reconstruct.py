import unittest

import numpy as np
from scipy.misc import imread

from virtualscanner.server.recon.drunck import reconstruct
from virtualscanner.utils import constants

RECON_ASSETS_PATH = constants.RECON_ASSETS_PATH
RECON_STATIC_SAVE_PATH = constants.COMS_PATH / 'coms_ui' / 'static' / 'recon' / 'outputs'


class MyTestCase(unittest.TestCase):
    def test_drunck_recon(self):
        """
        Unit test DRUNCK reconstruction of a 256x256 undersampled image.
        """
        aliased_filename, output_filename = reconstruct.main(img_path=RECON_ASSETS_PATH / 'utest_undersampled.jpg',
                                                             img_type='US')
        predicted_recon = imread(RECON_STATIC_SAVE_PATH / output_filename)
        utest_recon = imread(RECON_ASSETS_PATH / 'utest_recon.jpg')

        np.testing.assert_allclose(predicted_recon, utest_recon)


if __name__ == '__main__':
    unittest.main()
