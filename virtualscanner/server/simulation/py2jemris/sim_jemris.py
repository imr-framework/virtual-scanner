# Caller script that takes user inputs and performs a simulation using JEMRIS
from virtualscanner.server.simulation.py2jemris.seq2xml import seq2xml
from virtualscanner.server.simulation.py2jemris.sim2xml import sim2xml
from virtualscanner.server.simulation.py2jemris.recon_jemris import read_jemris_output
from virtualscanner.server.simulation.py2jemris.coil2xml import coil2xml
import subprocess
import tkinter as tk
from tkinter.filedialog import askopenfilename
from virtualscanner.utils import constants

import os

# Paths
PY2JEMRIS_SIM_PATH = constants.SERVER_SIM_BLOCH_PY2JEMRIS_PATH / 'sim'


def ask_for_sim_files():
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
    print("Simulating using JEMRIS ...")
    # Always rom from the py2jemris/sim directory
    if working_folder is None:
        working_folder = 'sim'
    os.chdir(working_folder)
    print(os.system('dir'))
    out = os.system('jemris simu.xml')
    print(out)
    # Find signal.h5
    signal_path = working_folder / 'signals.h5' # Return the absolute signal path here
    return signal_path

def sim_jemris(list_sim_files=None, working_folder=None):
    """Runs a JEMRIS MR simulation using given .xml and .h5 files
    #         based on custom file inputs.
    Output:
    -------
    signal : np.ndarray
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


    Mxy_out, M_vec_out, times_out = read_jemris_output(signal_path)
    output = {'Mxy': Mxy_out, "M_vec": M_vec_out, 'T': times_out}

    return output





if __name__ == '__main__':
    #['seq_xml', 'pht_h5', 'tx_xml', 'rx_xml', 'working_path'
    #output = sim_jemris()
  #  print(output)
   # sim2xml(seq="gre.xml", phantom="sample.h5", Tx="uniform.xml", Rx="uniform.xml",
  #        seq_name="Sequence", sample_name="Sample", out_folder_name="sim")
    list_sim_files = {'seq_xml': 'gre.xml', 'pht_h5': 'sample.h5', 'tx_xml':'uniform.xml',
                       'rx_xml': '8chheadcyl.xml'}
    out = sim_jemris(list_sim_files)
    print(out)