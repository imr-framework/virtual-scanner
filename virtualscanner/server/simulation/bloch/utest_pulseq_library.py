import unittest
import virtualscanner.server.simulation.bloch.pulseq_library as psl
from pypulseq.Sequence.sequence import Sequence
from virtualscanner.utils import constants
import time
import multiprocessing as mp
import virtualscanner.server.simulation.bloch.caller_script_blochsim as caller
import virtualscanner.server.simulation.bloch.pulseq_bloch_simulator as simulator
import virtualscanner.server.simulation.bloch.pulseq_blochsim_methods as blcsim
import numpy as np
import virtualscanner.server.simulation.bloch.phantom as pht

# Shared sequence parameters
FOV = 0.256
N = 15
thk = 5e-3
FA = 90
TR = 0.5
TE = 0.05
TI = 0.02
enc_ortho = 'xyz'
enc_oblique = [(2, 1, 0), (-1, 2, 0), (0, 0, 1)]


class TestPulseqGeneration(unittest.TestCase):
    def test_pulseq_gre_orthogonal(self):
        enc = 'xyz'
        # Make a GRE sequence
        seq = psl.make_pulseq_gre(fov=FOV,n=N,thk=thk,fa=FA,tr=TR,te=TE,enc=enc_ortho,slice_locs=[0],write=False)
        self.assertIsInstance(seq, Sequence)
        self.assertNotEqual(len(seq.block_events),0)

        # Delete files
    def test_pulseq_gre_oblique(self):
        seq = psl.make_pulseq_gre_oblique(fov=FOV,n=N,thk=thk,fa=FA,tr=TR,te=TE,enc=enc_oblique,slice_locs=[0],write=False)
        self.assertIsInstance(seq, Sequence)
        self.assertNotEqual(len(seq.block_events), 0)

    def test_pulseq_se_orthogonal(self):
        seq = psl.make_pulseq_se(fov=FOV,n=N,thk=thk,fa=FA,tr=TR,te=TE,enc=enc_ortho,slice_locs=[0],write=False)
        self.assertIsInstance(seq, Sequence)
        self.assertNotEqual(len(seq.block_events), 0)

    def test_pulseq_se_oblique(self):
        seq = psl.make_pulseq_se_oblique(fov=FOV,n=N,thk=thk,fa=FA,tr=TR,te=TE,enc=enc_oblique,slice_locs=[0],write=False)
        self.assertIsInstance(seq, Sequence)
        self.assertNotEqual(len(seq.block_events), 0)

    def test_pulseq_irse_orthogonal(self):
        seq = psl.make_pulseq_irse(fov=FOV,n=N,thk=thk,fa=FA,tr=TR,te=TE,ti=TI,enc=enc_ortho,slice_locs=[0],write=False)
        self.assertIsInstance(seq, Sequence)
        self.assertNotEqual(len(seq.block_events), 0)

    def test_pulseq_irse_oblique(self):
        seq = psl.make_pulseq_irse_oblique(fov=FOV,n=N,thk=thk,fa=FA,tr=TR,te=TE,ti=TI, enc=enc_oblique,slice_locs=[0],write=False)
        self.assertIsInstance(seq, Sequence)
        self.assertNotEqual(len(seq.block_events), 0)

    def test_pulseq_epi_se(self):
        seq, _, _ = psl.make_pulseq_epi_oblique(fov=FOV,n=N,thk=thk,fa=FA,tr=TR,te=TE,enc=enc_oblique,slice_locs=[0],write=False,
                                          echo_type='se', n_shots=1, seg_type='blocked')
        self.assertIsInstance(seq, Sequence)
        self.assertNotEqual(len(seq.block_events), 0)
    def test_pulseq_epi_gre(self):
        seq, _, _ = psl.make_pulseq_epi_oblique(fov=FOV,n=N,thk=thk,fa=FA,tr=TR,te=TE,enc=enc_oblique,slice_locs=[0],write=False,
                                          echo_type='gre', n_shots=1, seg_type='blocked')
        self.assertIsInstance(seq, Sequence)
        self.assertNotEqual(len(seq.block_events), 0)

def sim_sequence(seq):
    nn = 15
    phantom = pht.makeCylindricalPhantom(dim=2, dir='z', loc=0, n=nn)
    # Time the code: Tic
    start_time = time.time()
    # Store seq info
    seq_info = blcsim.store_pulseq_commands(seq)
    # Get list of locations from phantom
    loc_ind_list = phantom.get_list_inds()
    # Initiate multiprocessing pool
    pool = mp.Pool(mp.cpu_count())
    # Parallel simulation
    df = 0
    results = pool.starmap_async(blcsim.sim_single_spingroup,
                                 [(loc_ind, df, phantom, seq_info) for loc_ind in loc_ind_list]).get()
    pool.close()
    # Add up signal across all SpinGroups
    signal = np.sum(results, axis=0)

    # Time the code: Toc
    print("Time used: %s seconds" % (time.time() - start_time))

    return signal

def recon(signal):
    Nf, Np = (N, N)
    Ns = 1
    im_mat = np.zeros((Nf, Np, Ns), dtype=complex)
    kspace = np.zeros((Nf, Np, Ns), dtype=complex)
    for v in range(Ns):
        kspace[:, :, v] = np.transpose(signal[v * Np:v * Np + Np])
        im_mat[:, :, v] = np.fft.fftshift(np.fft.ifft2(kspace[:, :, v]))

    return kspace, im_mat

def recon_epi(signal, ro_dirs, ro_order):
    # Multislice reconstruction
    Nf, Np = (N, N)
    Ns = 1
    im_mat = np.zeros((Nf, Np, Ns), dtype=complex)
    kspace = np.zeros((Nf, Np, Ns), dtype=complex)

    for v in range(Ns):  # For each slice
        slice_signal = signal[v * Np:v * Np + Np]
        # slice_signal[0::2] = np.fliplr(slice_signal[0::2]) # Reverses every other line because of EPI
        # Flip reversed readout lines
        for u in range(len(slice_signal)):
            if ro_dirs[u]:
                slice_signal[u] = np.flip(slice_signal[u])
        # Reorder readout lines (only for interleaved mode)
        if len(ro_order) != 0:
            #            print(np.round(slice_signal,2))
            #  print('reordering')
            slice_signal = slice_signal[ro_order]
        #  print(np.round(slice_signal,2))

        #        kspace[:, :, v] = np.transpose(my_signal[v * Np:v * Np + Np])
        kspace[:, :, v] = np.transpose(slice_signal)
        im_mat[:, :, v] = np.fft.fftshift(np.fft.ifft2(kspace[:, :, v]))

    return kspace, im_mat

class TestPulseqSimulation(unittest.TestCase):
    # Check that: images are correctly generated; assert allclose to pre-generated images
    def test_sim_gre(self):
        seq = psl.make_pulseq_gre(fov=FOV,n=N,thk=thk,fa=FA,tr=TR,te=TE,enc=enc_ortho,slice_locs=[0],write=False)
        signal = sim_sequence(seq)
        kspace, im_mat = recon(signal)
        # Check dimensions only now
        self.assertEqual(np.shape(kspace), (15,15,1))
        self.assertEqual(np.shape(im_mat), (15,15,1))


    def test_sim_se(self):
        seq = psl.make_pulseq_se(fov=FOV,n=N,thk=thk,fa=FA,tr=TR,te=TE,enc=enc_ortho,slice_locs=[0],write=False)
        signal = sim_sequence(seq)
        kspace, im_mat = recon(signal)
        # Check dimensions only now
        self.assertEqual(np.shape(kspace), (15, 15, 1))
        self.assertEqual(np.shape(im_mat), (15, 15, 1))

    def test_sim_irse(self):
        seq = psl.make_pulseq_irse(fov=FOV,n=N,thk=thk,fa=FA,tr=TR,te=TE,ti=TI, enc=enc_ortho,slice_locs=[0],write=False)
        signal = sim_sequence(seq)
        kspace, im_mat = recon(signal)
        # Check dimensions only now
        self.assertEqual(np.shape(kspace), (15, 15, 1))
        self.assertEqual(np.shape(im_mat), (15, 15, 1))


    def test_sim_epi(self):
        seq, ro_dirs, ro_order = psl.make_pulseq_epi_oblique(fov=FOV,n=N,thk=thk,fa=FA,tr=TR,te=TE,enc=enc_oblique,slice_locs=[0],write=False,
                                          echo_type='se', n_shots=1, seg_type='blocked')
        signal = sim_sequence(seq)
        kspace, im_mat = recon_epi(signal, ro_dirs, ro_order)
        # Check dimensions only now
        self.assertEqual(np.shape(kspace), (15, 15, 1))
        self.assertEqual(np.shape(im_mat), (15, 15, 1))





if __name__ == "__main__":
    unittest.main()
