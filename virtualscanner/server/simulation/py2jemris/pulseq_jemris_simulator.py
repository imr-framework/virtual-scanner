# Same role as pulseq_bloch_simulator.py except (1) It uses JEMRIS (2) It is run as a function, not a script

# INPUTS: sequence type, geometry parameters, contrast parameters,
#         phantom type (pre-set or custom), coil type (pre-set or custom) (Tx / Rx),
#         k-space trajectory (if noncartesian; flattened version (Nro_total, 3))
import time
import os
from virtualscanner.server.simulation.py2jemris.sim_jemris import sim_jemris
from virtualscanner.server.simulation.py2jemris.recon_jemris import read_jemris_output, recon_jemris
from virtualscanner.server.simulation.py2jemris.coil2xml import coil2xml
from virtualscanner.utils import constants
from virtualscanner.server.simulation.py2jemris.seq2xml import seq2xml
from pypulseq.Sequence.sequence import Sequence
import virtualscanner.server.simulation.bloch.phantom as pht
import numpy as np
import xml.etree.ElementTree as ET
from virtualscanner.server.simulation.bloch.pulseq_library import make_pulseq_se_oblique,\
    make_pulseq_gre_oblique, make_pulseq_irse_oblique
from scipy.io import savemat, loadmat

PY2JEMRIS_PATH = constants.SERVER_SIM_BLOCH_PY2JEMRIS_PATH

# This function is for simulating a .seq sequence you just coded
def simulate_pulseq_jemris(seq_path, phantom_info, coil_fov,
                           tx='uniform', rx='uniform', # TODO add input that includes sequence info for
                                                       # TODO      dimensioning the RO points into kspace
                           tx_maps=None, rx_maps=None, sim_name=None):
    if sim_name is None:
        sim_name = time.strftime("%Y%m%d%H%M%S")

    target_path = PY2JEMRIS_PATH / 'sim' / sim_name

    # Make target folder
    dir_str = f'{str(PY2JEMRIS_PATH)}\\sim\\{sim_name}'
    if not os.path.isdir(dir_str):
        os.system(f'mkdir {dir_str}')

    # Convert .seq to .xml
    seq = Sequence()
    seq.read(seq_path)
    seq_name = seq_path[seq_path.rfind('/')+1:seq_path.rfind('.seq')]
    seq2xml(seq, seq_name=seq_name, out_folder_name=str(target_path))

    # Make phantom and save as .h5 file
    pht_name = create_and_save_phantom(phantom_info, out_folder_name=target_path)


    # Make sure we have the tx/rx files
    tx_filename = tx + '.xml'
    rx_filename = rx + '.xml'

    # Save Tx as xml
    if tx == 'uniform':
        os.system(f'copy sim\\{tx}.xml {str(target_path)}')
    elif tx == 'custom' and tx_maps is not None:
        coil2xml(b1maps=tx_maps, fov=coil_fov, name='custom_tx', out_folder=target_path)
        tx_filename = 'custom_tx.xml'
    else:
        raise ValueError('Tx coil type not found')

    # save Rx as xml
    if rx == 'uniform':
        os.system(f'copy sim\\{rx}.xml sim\\{str(target_path)}')
    elif rx == 'custom' and rx_maps is not None:
        coil2xml(b1maps=rx_maps, fov=coil_fov, name='custom_rx', out_folder=target_path)
        rx_filename = 'custom_rx.xml'
    else:
        raise ValueError('Rx coil type not found')

    # Run simuluation in target folder
    list_sim_files = {'seq_xml': seq_name+'.xml', 'pht_h5': pht_name + '.h5', 'tx_xml': tx_filename,
                       'rx_xml': rx_filename}
    sim_output = sim_jemris(list_sim_files=list_sim_files, working_folder=target_path)

    return sim_output


# TODO
def create_and_save_phantom(phantom_info, out_folder_name):

    out_folder_name = str(out_folder_name)

    FOV = phantom_info['fov']
    N = phantom_info['N']
    pht_type = phantom_info['type']
    pht_dim = phantom_info['dim']
    pht_dir = phantom_info['dir']

    sim_phantom = 0

    if pht_type == 'spherical':
        print('Making spherical phantom')
        T1s = [1000]
        T2s = [100]
        PDs = [1]
        R = 0.8*FOV/2
        Rs = [R]
        if pht_dim == 3:
            sim_phantom = pht.makeSphericalPhantom(n=N, fov=FOV, T1s=T1s, T2s=T2s, PDs=PDs, radii=Rs)

        elif pht_dim == 2:
            sim_phantom = pht.makePlanarPhantom(n=N, fov=FOV, T1s=T1s, T2s=T2s, PDs=PDs, radii=Rs,
                                                dir=pht_dir, loc=0)
    elif pht_type == 'cylindrical':
        print("Making cylindrical phantom")
        sim_phantom = pht.makeCylindricalPhantom(dim=pht_dim, n=N, dir=pht_dir, loc=0)

    elif pht_type == 'custom':
        # Use a custom file!
        T1 = phantom_info['T1']
        T2 = phantom_info['T2']
        PD = phantom_info['PD']
        dr = phantom_info['dr']
        if 'dBmap' in phantom_info.keys():
            dBmap = phantom_info['dBmap']
        else:
            dBmap = 0

        sim_phantom = pht.Phantom(T1map=T1, T2map=T2, PDmap=PD, vsize=dr, dBmap=dBmap, loc=(0,0,0))

    else:
        raise ValueError("Phantom type non-existent!")

    # Save as h5
    sim_phantom.output_h5(out_folder_name, pht_type)

    return pht_type

if __name__ == '__main__':
    # Define the same phantom
    phantom_info = {'fov': 0.256, 'N': 16, 'type': 'cylindrical', 'dim':2, 'dir':'z'}
    sim_names = ['test0413_GRE', 'test0413_SE', 'test0413_IRSE']
    sps = ['gre_fov256mm_Nf15_Np15_TE50ms_TR200ms_FA90deg.seq',
           'se_fov256mm_Nf15_Np15_TE50ms_TR200ms_FA90deg.seq',
           'irse_fov256mm_Nf15_Np15_TI20ms_TE50ms_TR200ms_FA90deg.seq']
    # make_pulseq_irse_oblique(fov=0.256,n=15, thk=0.005, tr=0.2, te=0.05, ti=0.02, fa=90,
    #                          enc='xyz', slice_locs=[0], write=True)
    # make_pulseq_gre_oblique(fov=0.256,n=15, thk=0.005, tr=0.2, te=0.05, fa=90,
    #                          enc='xyz', slice_locs=[0], write=True)
    # make_pulseq_se_oblique(fov=0.256,n=15, thk=0.005, tr=0.2, te=0.05, fa=90,
    #                          enc='xyz', slice_locs=[0], write=True)
    simulate_pulseq_jemris(seq_path=sps[0], phantom_info=phantom_info, sim_name=sim_names[0],
                               coil_fov=0.256)
    kk, im, images = recon_jemris(file='sim/' + sim_names[0] + '/signals.h5', dims=[15,15])
    savemat('sim/'+sim_names[0]+'/output.mat', {'images': images, 'kspace': kk, 'imspace': im})
