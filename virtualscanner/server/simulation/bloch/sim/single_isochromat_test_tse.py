import multiprocessing as mp
import time
#import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
from pypulseq.Sequence.sequence import Sequence
import virtualscanner.server.simulation.bloch.phantom as pht
import virtualscanner.server.simulation.bloch.pulseq_blochsim_methods as blcsim
import virtualscanner.server.simulation.bloch.pulseq_library as psl
from virtualscanner.server.simulation.bloch.phantom_acr import make_phantom_acr, make_phantom_circle
import virtualscanner.server.simulation.bloch.spingroup_ps as sg
from scipy.io import savemat
import matplotlib.pyplot as plt


if __name__ == '__main__':

    myseq = Sequence()
    myseq.read('seq_validation_files/tse32_zero.seq')

    seq_info =  blcsim.store_pulseq_commands(myseq)
    print(seq_info['commands'][35])
    print((seq_info['params'][35][4]))

    seq_info = blcsim.store_pulseq_commands(myseq)

    isc = sg.NumSolverSpinGroup(loc=(0,0,0), pdt1t2=(1,0.5,0.5),df=0)
    m_store = blcsim.apply_pulseq_commands(isc,seq_info,store_m=True)

    savemat('seq_validation_files/isc_signal_cancel-2-ADC_new.mat',{'signal':isc.signal, 'm':m_store})