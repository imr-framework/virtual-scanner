import subprocess
import numpy as np





#n = 5
#siminfo = {'pht_type':'spherical', 'pht_dim': '3', 'n_ph':str(n), 'fov_ph': '0.256', 'dir_ph':'z',
 #              'seq_type':'gre', 'thk':str(0.256/n), 'n':str(n), 'fov':'0.256', 'enc':'xyz',
  #         'tr':'5','te':'0.1','ti':'0.02','fa':'90','num_slices':'3','slice_gap':'0','b0map':'0'}

siminfo1 = {'formName':'acq','selectedSeq':'GRE','TR':'1000','TE':'500','FA':'90',
            'sl-orient':'axial','thck':str(240/5),'slicenum':'3','freq':'x','ph':'y','bw':'3','Nx':'5','Ny':'5','FOVx':'240','FOVy':'240'}

siminfo2 = {'formName':'acq','selectedSeq':'SE','TR':'1000','TE':'500','FA':'90',
            'sl-orient':'axial','thck':'2','slicenum':'3','freq':'x','ph':'y','bw':'3','Nx':'5','Ny':'5','FOVx':'240','FOVy':'240'}

siminfo3 = {'formName':'acq','selectedSeq':'GRE','IRSE':'ON','TR':'1000','TE':'500','TI':'20','FA':'90',
            'sl-orient':'axial','thck':'2','slicenum':'3','freq':'x','ph':'y','bw':'3','Nx':'5','Ny':'5','FOVx':'240','FOVy':'240'}



#TR, TE and TI in ms
#FA in deg
#slice thickness in mm


# Parse arguments from dictionary (str to str!)
def run_blochsim(siminfo):
    # Parse seq info
    orient_dict = {'sagittal':'x','coronal':'y','axial':'z'}
    tr = str(float(siminfo['TR'])*1e-3)
    te = str(float(siminfo['TE'])*1e-3)
    if 'TI' in siminfo.keys():
        ti = str(float(siminfo['TI'])*1e-3)
    else:
        ti = '0'


    fa = siminfo['FA']
    n = siminfo['Nx']
    fov = str(float(siminfo['FOVx'])*1e-3)
    enc = siminfo['freq']+siminfo['ph']+ orient_dict[siminfo['sl-orient']]
    seq_type = siminfo['selectedSeq'].lower()
    if seq_type == 'se' and siminfo['IRSE'] == 'ON':
        seq_type = 'irse'
    thk = str(float(siminfo['thck'])*1e-3)
    slice_gap = '0'
    num_slices = siminfo['slicenum']

    # Parse phantom info  Now, just default
    pht_type = 'spherical'
    pht_dim = '3'
    n_ph = '5'
    fov_ph = '0.240'
    dir_ph = 'z'

    subprocess.run(['python.exe',
                     #'-m','cProfile','-o','profiling_results', #<- for profiling code
                    'pulseq_bloch_simulator.py',
                     pht_type,pht_dim,n_ph,fov_ph, # pht_type, dim, Nph, FOVph(m)
                     '1','1','1',# PDs (a.u.) - default values
                    '2','1','0.5', # T1s (s) - defualt values
                    '0.1','0.15','0.25', #T2s (s) - default values
                    dir_ph,  # phantom slice direction (only for 2D)
                    seq_type, num_slices, thk, slice_gap, # sequence type, #slices, slice thk, slice gap
                    n,fov,enc,# N, FOV(m), enc
                    tr,te,ti,fa,'0'])# TR(s), TE(s), TI(s), FA(deg), type of b0 map (0 for now)
    # load saved data (reconstruct how? many slices - how is it stored?)


if __name__ == "__main__":
    run_blochsim(siminfo1)