# Copyright of the Board of Trustees of Columbia University in the City of New York

import subprocess

from virtualscanner.utils import constants

SERVER_SIM_BLOCH_PATH = constants.SERVER_SIM_BLOCH_PATH

def run_blochsim(seqinfo,phtinfo,pat_id):
    """Caller function that runs Bloch simulation

    This function parses parameters for the requested simulation
    and then executes the main simulation script, pulseq_bloch_simulator.py,
    passing on the parameters

    Parameters
    ----------
    seqinfo : dict
        Acquire page payload
    phtinfo : dict
        Register page payload
    pat_id : str
        Patient ID of current session

    Returns
    -------
    num : int=1

    """

    # Parse seq info
    orient_dict = {'sagittal':'x','coronal':'y','axial':'z'}
    tr = str(float(seqinfo['TR'])*1e-3)
    te = str(float(seqinfo['TE'])*1e-3)
    if 'TI' in seqinfo.keys():
        ti = str(float(seqinfo['TI'])*1e-3)
    else:
        ti = '0'
    fa = seqinfo['FA']
    n = seqinfo['Nx']
    fov = str(float(seqinfo['FOVx'])*1e-3)
    enc = seqinfo['freq']+seqinfo['ph']+ orient_dict[seqinfo['sl-orient']]
    seq_type = seqinfo['selectedSeq'].lower()
    if seq_type == 'se' and 'IRSE' in seqinfo.keys():
        print("IRSE exists")
        if seqinfo['IRSE'] == 'on':
            print("irse on")
            seq_type = 'irse'
    thk = str(float(seqinfo['thck'])*1e-3)
    slice_gap = '0'
    num_slices = seqinfo['slicenum']

    # Parse phantom info  Now, just default
    if phtinfo == 'Numerical':
        pht_type = 'cylindrical'
        pht_dim = '2'
        n_ph = '15'
        fov_ph = '0.240'
        dir_ph = enc[2]

    else:
        print('Phantom type: '+phtinfo+" not supported")


    subprocess.run(['python',
                    str(SERVER_SIM_BLOCH_PATH / 'pulseq_bloch_simulator.py'),
                     pat_id, # patient id
                     pht_type,pht_dim,n_ph,fov_ph, # pht_type, dim, Nph, FOVph(m)
                     '1','1','1',# PDs (a.u.) - default values
                    '2','1','0.5', # T1s (s) - defualt values
                    '0.1','0.15','0.25', #T2s (s) - default values
                    dir_ph,  # phantom slice direction (only for 2D)
                    seq_type, num_slices, thk, slice_gap,
                    n,fov,enc,# N, FOV(m), enc
                    tr,te,ti,fa,'0'])# TR(s), TE(s), TI(s), FA(deg), type of b0 map (0 for now)

    return 1
