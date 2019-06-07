import subprocess
import numpy as np
import os

#n = 5
#siminfo = {'pht_type':'spherical', 'pht_dim': '3', 'n_ph':str(n), 'fov_ph': '0.256', 'dir_ph':'z',
 #              'seq_type':'gre', 'thk':str(0.256/n), 'n':str(n), 'fov':'0.256', 'enc':'xyz',
  #         'tr':'5','te':'0.1','ti':'0.02','fa':'90','num_slices':'3','slice_gap':'0','b0map':'0'}

#TR, TE and TI in ms
#FA in deg
#slice thickness in mm


# Parse arguments from dictionary (str to str!)
def run_blochsim(seqinfo,phtinfo,pat_id):
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
        #pht_type = 'spherical'
        pht_type = 'cylindrical'
        pht_dim = '2'
        n_ph = '15'
        fov_ph = '0.240'
        dir_ph = enc[2]
        #dir_ph = seqinfo['enc'][2] # not used for 3D though ## TODO


    else:
        print('Phantom type: '+phtinfo+" not supported")

 #   else:
  #      pht_type = 'spherical'
   #     pht_dim = '3'
    #    n_ph = '5'
     #   fov_ph = '0.240'
      #  dir_ph = 'z'


    subprocess.run(['python', #This is specific to windows
                     #'-m','cProfile','-o','profiling_results', #<- for profiling code
                    os.path.join(os.path.dirname(os.path.realpath(__file__)),'pulseq_bloch_simulator.py'),
                     pat_id, # patient id
                     pht_type,pht_dim,n_ph,fov_ph, # pht_type, dim, Nph, FOVph(m)
                     '1','1','1',# PDs (a.u.) - default values
                    '2','1','0.5', # T1s (s) - defualt values
                    '0.1','0.15','0.25', #T2s (s) - default values
                    dir_ph,  # phantom slice direction (only for 2D)
                    seq_type, num_slices, thk, slice_gap, # sequence type, #slices, slice thk, slice gap
                    n,fov,enc,# N, FOV(m), enc
                    tr,te,ti,fa,'0'])# TR(s), TE(s), TI(s), FA(deg), type of b0 map (0 for now)
    # load saved data (reconstruct how? many slices - how is it stored?)

    return 1
