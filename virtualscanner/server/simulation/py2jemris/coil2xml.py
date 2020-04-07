# Take user inputs and convert them into either coil2xml or

import xml.etree.ElementTree as ET

import h5py
import matplotlib.pyplot as plt
import numpy as np



# TODO: consider making Coil a class!

def circle_of_loops_2d(R, n_loop, diameter):
    coil_design = {}

    return coil_design


def coil2xml(b1maps=None, coil_design=None, fov=256, name='coils', out_folder=''):
    """
    Inputs
    ------
    b1map : list, optional
        List of np.ndarray (dtype='complex') maps for all channels
    coil_design : dict, optional
        Dictionary containing information on coil design (see documentation)
    fov : float
        Field-of-view of coil in mm
    name : str, optional
        Name of generated .xml file
    working_folder : str, optional


    Returns
    -------
    None

    """
    if b1maps is None and coil_design is None:
        raise ValueError("One of b1map and coil_design must be provided")

    # b1 map case
    if b1maps is not None:

        root = ET.Element('CoilArray')

        for ch in range(len(b1maps)): # for each channel
            # Check dimensions
            if len(b1maps[ch].shape) < 2 or len(b1maps[ch].shape) > 3 \
                    or max(b1maps[ch].shape) != min(b1maps[ch].shape):
                raise ValueError("b1map must be a 2D square or 3D cubic array: \
                                  all sides must be equal")

            N = b1maps[ch].shape[0]
            dim = len(b1maps[ch].shape)

            b1_magnitude = np.absolute(b1maps[ch])
            b1_phase = np.angle(b1maps[ch])

            # Make h5 file
            coil_h5_path = name + f'_ch{ch+1}.h5'
            coil = h5py.File(out_folder + '/' + coil_h5_path, 'a')
            if 'maps' in coil.keys():
                del coil['maps']
            maps = coil.create_group('maps')
            magnitude = maps.create_dataset('magnitude',b1_magnitude.shape,dtype='f')
            phase = maps.create_dataset('phase',b1_phase.shape,dtype='f')


            # Set h5 file contents
            if dim == 2:
                magnitude[:,:] = b1_magnitude
                phase[:,:] = b1_phase

            elif dim == 3:
                print('3d!')
                magnitude[:,:,:] = b1_magnitude
                phase[:,:,:] = b1_phase


            coil.close()

            # Add corresponding coil to .xml tree
            externalcoil = ET.SubElement(root, "EXTERNALCOIL")
            externalcoil.set("Points",str(N))
            externalcoil.set("Name",f"C{ch+1}")
            externalcoil.set("Filename",coil_h5_path)
            externalcoil.set("Extent",str(fov)) # fov is in mm
            externalcoil.set("Dim",str(dim))

        coil_tree = ET.ElementTree(root)
        coil_tree.write(out_folder + '/' + name + '.xml')


    # TODO Biot-Savart loop based designs
    # coil design case
    #elif coil_design is not None:
    #    root = ET.Element("CoilArray")
    #    for coil_key in coil_design.keys():
    #        coil_params = coil_design[coil_key]

    #    # for each entry, add a loop




if __name__ == '__main__':
    # a = h5py.File('coil2xml/sensmaps.h5','a')
    # print(a['maps'].keys())
    # print(a['maps']['magnitude'].keys())
    # map_mag = a['maps']['magnitude']
    # map_phase = a['maps']['phase']
    #
    #
    # plt.figure(1)
    #
    # for ch in range(8):
    #     plt.subplot(2,8,ch+1)
    #     plt.imshow(map_mag[f'0{ch}'][()])
    #     plt.subplot(2,8,ch+9)
    #     plt.imshow(map_phase[f'0{ch}'][()])
    #
    # plt.show()

    b1 = np.ones((32,32))
    XY = np.meshgrid(np.linspace(0,1,32), np.linspace(0,1,32))
    X = XY[0]
    Y = XY[1]
    b1 = np.sqrt(X**2 + Y**2)
    plt.imshow(b1)
    plt.show()
    coil2xml(b1maps=[b1], fov=200, name='test_coil', out_folder='coil2xml')