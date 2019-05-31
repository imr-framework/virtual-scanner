import subprocess



siminfo = {'formName':'acq','selectedSeq':'GRE','TR':'1000','TE':'500','FA':'90','TI':'50',
            'sl-orient':'axial','thck':str(240/16),'slicenum':'1','freq':'x','ph':'y','bw':'3','Nx':'15','Ny':'15','FOVx':'240','FOVy':'240'}



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
    pht_type = 'cylindrical'
    n_ph = '15'
    pht_dim = '2'
    fov_ph = '0.240'
    dir_ph = 'z'
#


    subprocess.run(['python.exe',
                     #'-m','cProfile','-o','profiling_results', #<- for profiling code
                    'pulseq_bloch_simulator.py','11037',
                     pht_type,pht_dim,n_ph,fov_ph, # pht_type, dim, Nph, FOVph(m)
                     '1','1','1',# PDs (a.u.) - default values
                    '2','1','0.5', # T1s (s) - defualt values
                    '0.1','0.15','0.25', #T2s (s) - default values
                    dir_ph,  # phantom slice direction (only for 2D)
                    seq_type, num_slices, thk, slice_gap, # sequence type, #slices, slice thk, slice gap
                    n,fov,enc,# N, FOV(m), enc
                    tr,te,ti,fa,'0'])# TR(s), TE(s), TI(s), FA(deg), type of b0 map (0 for now)


if __name__ == "__main__":
    run_blochsim(siminfo)