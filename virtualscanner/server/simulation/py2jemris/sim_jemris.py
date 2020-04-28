# Caller script for executing a simulation with JEMRIS (prior installation required)
# Gehua Tong, April 2020




from virtualscanner.server.simulation.py2jemris.seq2xml import seq2xml
from virtualscanner.server.simulation.py2jemris.sim2xml import sim2xml
from virtualscanner.server.simulation.py2jemris.recon_jemris import read_jemris_output
from virtualscanner.server.simulation.py2jemris.coil2xml import coil2xml
import subprocess
import tkinter as tk
from tkinter.filedialog import askopenfilename
from virtualscanner.utils import constants
import h5py
import os

# Paths
PY2JEMRIS_SIM_PATH = constants.SERVER_SIM_BLOCH_PY2JEMRIS_PATH / 'sim'
from scipy.io import savemat
import time

def ask_for_sim_files():
    """Helper function for sim_jemris;
       Asks the user for simulation files through file system selection

    Returns
    -------
    files_list : list
        A dictionary indicating paths to the files required to construct simu.xml

    """
    files_list = {}
    names = ['seq_xml', 'pht_h5', 'tx_xml', 'rx_xml']
    prompt_list = ['sequence file (.xml)', 'phantom file (.h5)','Tx file (.xml)', 'Rx file (.xml)']

    for u in range(len(prompt_list)-1):
        print(f"Pick your {prompt_list[u]}.")
        tk.Tk().withdraw()
        filename = askopenfilename()
        files_list[names[u]] = filename

    return files_list



def run_jemris(working_folder = None):
    """Runs JEMRIS simulation on system command line
       Assumes that the working folder contains all required files and
               that JEMRIS is installed and added to PATH on the operating system
       Simply, the command "jemris simu.xml" is run and the path to signals.h5 is returned

    Inputs
    ------
    working_folder : str
        Working folder where the simulation is performed

    Returns
    -------
    signal path : str or pathlib Path object
        Path to JEMRIS simulation output data file (this file is always called signals.h5)

    """
    print("Simulating using JEMRIS ...")
    # Always rom from the py2jemris/sim directory
    if working_folder is None:
        working_folder = 'sim'
    original_wd = os.getcwd()
    os.chdir(working_folder)
    print(os.system('dir'))
    out = os.system('jemris simu.xml')
    print(out)
    os.chdir(original_wd)
    # Find signal.h5
    if isinstance(working_folder, str):
        signal_path = working_folder + '/signals.h5'
    else:
        signal_path = working_folder / 'signals.h5' # Return the absolute signal path here


    return signal_path

def sim_jemris(list_sim_files=None, working_folder=None):
    """Runs a JEMRIS MR simulation using given .xml and .h5 files
              based on custom file inputs. Returns complex signal data.
    Inputs
    ------
    list_sim_files : dict
        Dictionary of paths to relevant simulation files
    working_folder : str
        Working folder where the simulation is performed


    Returns
    -------
    output : dict
        Complex signal data with 3 fields
        'Mxy' : Complex representation of transverse magnetization
        'M_vec' : 3D representation of magnetization (Mx, My, Mz)
        'T' : Timing of readout points

    """

    # Use interactive option if there is no dictionary input
    all_files_exist = False
    while not all_files_exist:
        try:
            seq_xml = list_sim_files['seq_xml']
            pht_h5 = list_sim_files['pht_h5']
            tx_xml = list_sim_files['tx_xml']
            rx_xml = list_sim_files['rx_xml']

            all_files_exist = True
        except:
            list_sim_files = ask_for_sim_files()

    # Extract sequence and phantom name
    seq_name = seq_xml[seq_xml.rfind('/')+1:seq_xml.rfind('.xml')]
    pht_name = pht_h5[pht_h5.rfind('/')+1:pht_h5.rfind('.h5')]


    # Make simu.xml
    sim2xml(sim_name='simu', seq=seq_xml, phantom=pht_h5, Tx=tx_xml, Rx=rx_xml,
                        seq_name=seq_name, sample_name=pht_name, out_folder_name=str(working_folder))
    # Rum JEMRIS on command line
    signal_path = run_jemris(working_folder)
    print(signal_path)

    file_discovered = False
    print((os.path.abspath(signal_path)))
    while not file_discovered:
        file_discovered = os.path.exists(os.path.abspath(signal_path))
        print(file_discovered)
        time.sleep(2)

    Mxy_out, M_vec_out, times_out = read_jemris_output(signal_path)
    output = {'Mxy': Mxy_out, "M_vec": M_vec_out, 'T': times_out}

    return output



from virtualscanner.server.simulation.py2jemris.recon_jemris import *



if __name__ == '__main__':
    #['seq_xml', 'pht_h5', 'tx_xml', 'rx_xml', 'working_path'
    #output = sim_jemris()
  #  print(output)
   # sim2xml(seq="gre.xml", phantom="sample.h5", Tx="uniform.xml", Rx="uniform.xml",
  #        seq_name="Sequence", sample_name="Sample", out_folder_name="sim")


    # "Sim test" April 17 for seq2xml

    # First, sim using original gre
    list_sim_orig = {'seq_xml': 'gre_test_0417.xml', 'pht_h5': 'cylindrical.h5', 'tx_xml':'uniform.xml',
                       'rx_xml': 'uniform.xml'}
    out = sim_jemris(list_sim_orig, working_folder = 'sim/testfolder')
    # Save output in file

    # Second, use twice converted (.xml output of seq2xml)
    #list_sim_twice = {'seq_xml': 'gre_test_0417_twice.xml', 'pht_h5': 'cylindrical.h5', 'tx_xml':'uniform.xml',
  #                     'rx_xml': 'uniform.xml'}
  #  out = sim_jemris(list_sim_twice, working_folder = 'sim/testfolder')

    # Save output in file

    #kspace, imspace, images = recon_jemris('sim/testfolder/signals1.h5', [32,32])
    #savemat('sim/testfolder/data_orig.mat', {'kspace':kspace, 'imspace':imspace,'images':images})

