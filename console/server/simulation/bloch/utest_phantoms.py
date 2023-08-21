import unittest
import virtualscanner.server.simulation.bloch.phantom as pht
import numpy as np
from virtualscanner.utils import constants

class TestPhantoms(unittest.TestCase):
    def test_phantom_basic(self):
        T1s = np.reshape(np.linspace(0.1,2,8),(2,2,2))
        T2s = np.reshape(np.linspace(5e-3,0.5,8),(2,2,2))
        PDs = np.ones((2,2,2))
        v = 0.01 # 1 cm
        phantom = pht.Phantom(T1map=T1s, T2map=T2s, PDmap=PDs, vsize=v)

        np.testing.assert_array_equal(phantom.fov, np.array([0.02,0.02,0.02]))
        np.testing.assert_array_equal(phantom.Xs, np.array([-5e-3,5e-3]))
        np.testing.assert_array_equal(phantom.Ys, np.array([-5e-3,5e-3]))
        np.testing.assert_array_equal(phantom.Zs,  np.array([-5e-3,5e-3]))


    def test_phantom_DTT(self):
        tmap = np.reshape(np.arange(8),(2,2,2))
        params = {0:(1,1,0.5),1:(0.9,1,0.5),2:(1,0.5,0.5),3:(1,2,0.5),
                  4:(1,1,0.2),5:(1,1,0.05),6:(0,0,0),7:(1,0,0)}
        v = 0.01

        phantom = pht.DTTPhantom(type_map=tmap, type_params=params, vsize=v)


        np.testing.assert_array_equal(phantom.PDmap,np.reshape([1,0.9,1,1,1,1,0,1],(2,2,2)))
        np.testing.assert_array_equal(phantom.T1map,np.reshape([1,1,0.5,2,1,1,0,0],(2,2,2)))
        np.testing.assert_array_equal(phantom.T2map,np.reshape([0.5,0.5,0.5,0.5,0.2,0.05,0,0],(2,2,2)))


    def test_phantom_cylinder(self):
        # Compare to pre-made phantom (check size and values)
        phantom = pht.makeCylindricalPhantom()

        test_data_path = constants.SERVER_SIM_BLOCH_PATH / 'test_data' / 'test_cylindrical_phantom_maps.npy'
        maps = np.load(test_data_path)


        self.assertIsInstance(phantom, pht.DTTPhantom)
        np.testing.assert_allclose(phantom.PDmap, maps[:,:,:,0])
        np.testing.assert_allclose(phantom.T1map, maps[:,:,:,1])
        np.testing.assert_allclose(phantom.T2map, maps[:,:,:,2])



if __name__ == "__main__":
    unittest.main()
