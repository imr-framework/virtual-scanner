# Demonstrates usage of py2jemris functionalities
# May be used for quick testing
# Gehua Tong
# May 18, 2020

from virtualscanner.server.simulation.py2jemris.coil2xml import coil2xml
from virtualscanner.server.simulation.py2jemris.seq2xml import seq2xml
from virtualscanner.server.simulation.py2jemris.sim_jemris import sim_jemris
from virtualscanner.server.simulation.py2jemris.pulseq_jemris_simulator import simulate_pulseq_jemris, create_and_save_phantom
from virtualscanner.server.simulation.py2jemris.recon_jemris import recon_jemris
import virtualscanner.server.simulation.bloch.phantom as pht
from virtualscanner.server.simulation.bloch.pulseq_library import make_pulseq_irse, make_pulseq_se_oblique
import numpy as np
import matplotlib.pyplot as plt
from pypulseq.Sequence.sequence import Sequence
from scipy.io import loadmat, savemat
from virtualscanner.utils.constants import SERVER_SIM_BLOCH_PY2JEMRIS_PATH
import os
import h5py


utest_path = str(SERVER_SIM_BLOCH_PY2JEMRIS_PATH / 'sim' / 'utest_outputs')
sim_path = str(SERVER_SIM_BLOCH_PY2JEMRIS_PATH / 'sim')


def utest_coil2xml():
    # Example on using coil2xml
    # Generate coil using B1 maps and plot
    # 4 channels with different B1 maps

    b1 = np.ones((32,32))
    XY = np.meshgrid(np.linspace(0,1,32), np.linspace(0,1,32))
    X = XY[0]
    Y = XY[1]

    # Define coil sensitivity maps (complex arrays, in general)
    b1_ch1 = np.sqrt(X**2 + Y**2)
    b1_ch2 = np.rot90(b1_ch1)
    b1_ch3 = np.rot90(b1_ch2)
    b1_ch4 = np.rot90(b1_ch3)

    coil2xml(b1maps=[b1_ch1, b1_ch2, b1_ch3, b1_ch4], fov=200, name='test_coil', out_folder=utest_path)

    # Generate sensmaps.h5 using JEMRIS command
    os.chdir(utest_path)
    print(os.system('dir'))
    out = os.system('jemris test_coil.xml')
    print(out)


    # Load sensmaps.h5 and plot coil
    a = h5py.File(utest_path + '/sensmaps.h5', 'r')
    maps_magnitude = a['maps/magnitude']
    maps_phase = a['maps/phase']
    plt.figure(1)
    plt.title('Coil sensitivity maps')
    for u in range(4):
        plt.subplot(2,4,u+1)
        plt.gray()
        plt.imshow(maps_magnitude[f'0{u}'])
        plt.title(f'Magnitude Ch #{u+1}')
        plt.subplot(2,4,u+5)
        plt.gray()
        plt.imshow(maps_phase[f'0{u}'])
        plt.title(f'Phase Ch #{u+1}')
    plt.show()
    return

def utest_seq2xml():
    # Make a sequence
    seq = make_pulseq_irse(fov=0.256, n=16, thk=0.01, fa=15, tr=150, te=30, ti=10,
                           enc='xyz', slice_locs=None, write=False)

    # Convert to .xml format
    seq2xml(seq, seq_name='irse16_pulseq', out_folder=utest_path)



    # Use JEMRIS to generate sequence diagrams from .xml sequence
    os.chdir(utest_path)
    print(os.system('dir'))
    out = os.system(f'jemris -x -d id=1 -f irse16_pulseq irse16_pulseq.xml')
    print(out)

    # Read sequence diagram and plot
    data = h5py.File(utest_path + '/irse16_pulseq.h5','r')
    diag = data['seqdiag']

    t = diag['T']
    gx = diag['GX']
    gy = diag['GY']
    gz = diag['GZ']
    rxp = diag['RXP']
    txm = diag['TXM']
    txp = diag['TXP']

    ylist = [txm, txp, gx, gy, gz, rxp]
    title_list = ['RF Tx magnitude', 'RF Tx phase', 'Gx', 'Gy', 'Gz', 'RF Rx phase']
    styles = ['r-', 'g-', 'k-', 'k-', 'k-', 'bx']
    plt.figure(1)
    for v in range(6):
        plt.subplot(6,1,v+1)
        plt.plot(t, ylist[v], styles[v])
        plt.title(title_list[v])
        plt.xlabel('Time')

    plt.show()


    return

def utest_sim_jemris():
    # Copy helping files in
    os.chdir(sim_path)
    out = os.system('copy uniform.xml utest_outputs')
    print(out)
    utest_phantom_output_h5()

    list_sim_orig = {'seq_xml': 'gre32.xml', 'pht_h5': 'cylindrical.h5', 'tx_xml':'uniform.xml',
                       'rx_xml': 'uniform.xml'}
    out = sim_jemris(list_sim_orig, working_folder = utest_path)
    os.chdir(utest_path)
    savemat('data32_orig.mat',out)
    print('Data is saved in py2jemris/sim/utest_outputs/data32_orig.mat')
    return

def utest_pulseq_sim():
    # TODO this !
    # Demonstrates simulation pipeline using pulseq inputs

    # Define the same phantom
    phantom_info = {'fov': 0.256, 'N': 15, 'type': 'cylindrical', 'dim': 2, 'dir': 'z'}
    sps =  'se_fov256mm_Nf15_Np15_TE50ms_TR200ms_FA90deg.seq'
    sim_name = 'utest_outputs'
    # Make sequence
    os.chdir(utest_path)
    make_pulseq_se_oblique(fov=0.256,n=15, thk=0.005, tr=0.2, te=0.05, fa=90,
                              enc='xyz', slice_locs=[0], write=True)


    os.chdir(str(SERVER_SIM_BLOCH_PY2JEMRIS_PATH))
    simulate_pulseq_jemris(seq_path=sps, phantom_info=phantom_info, sim_name=sim_name,
                           coil_fov=0.256)

    kk, im, images = recon_jemris(file='sim/' + sim_name + '/signals.h5', dims=[15, 15])

    savemat('sim/' + sim_name + '/utest_pulseq_sim_output.mat', {'images': images, 'kspace': kk, 'imspace': im})
    print('Simulation result is in py2jemris/sim/utest_outputs/utest_pulseq_sim_output.mat')

    # Plot results
    plt.figure(1)
    plt.gray()
    plt.imshow(np.squeeze(images))
    plt.show()

    return

def utest_phantom_output_h5():
    # Creates a virtual scanner phantom and save it as an .h5 file (per JEMRIS standard)
    phantom_info = {'fov': 0.256, 'N': 32, 'type': 'cylindrical', 'dim': 2, 'dir': 'z'}
    create_and_save_phantom(phantom_info, out_folder=utest_path)
    return


if __name__ == '__main__':

    # Run all "utests"
    utest_coil2xml() # Converts B1 map into .h5 and .xml files for JEMRIS
    utest_phantom_output_h5() # Makes a virtual scanner phantom and converts it into .h5 format for JEMRIS
    utest_seq2xml() # Makes a pypulseq sequence and converts it into .xml and .h5 files for JEMRIS
    utest_sim_jemris() # Calls JEMRIS on command line using pre-made files
    utest_pulseq_sim() # Calls pipeline (INPUT: seq + phantom info + FOV ; OUTPUT: complex image space & k-space, images)
