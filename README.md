# Virtual Scanner
Virtual Scanner is an end-to-end hybrid MR simulator/console designed to be:
easily accessible, modular, and supported by open-source standards. 

The project is a response to the [ISMRM 2019 Junior Fellow Challenge (Africa)](https://www.ismrm.org/2019-junior-fellow-challenge/africa/).

Virtual Scanner consists of two modes: in Standard Mode, a console-like GUI allows users to perform virtual scans and conduct basic analysis; in Advanced Mode, modular simulation/analysis of the entire MR signal chain may be performed.  

## Standard Mode
### Register
The Register page allows you to choose a phantom for simulation. Its format is similar to the form for entering information of the subject when conducting real scans. Choose the "Numerical" phantom for all simulations now. 

### Acquire
The Acquire page allows the user to choose either a Gradient Echo (GRE) or a Spin Echo (SE, with optional inversion recovery) sequence, enter the parameters, and simulate them on a cylindrical phantom ("Numerical") containing balls with different T1, T2, and PD values. 

### Analyze
The Analyze page allows the user to load a series of data acquired for T1 or T2 mapping and conduct curve fitting to obtain T1 and T2 maps. In addition, it can detect circles on the maps, a common feature of real MR phantoms.

## Advanced Mode
### Tx (RF Transmit)
The Tx page allows one to calculate and plot SAR from pulseq .seq files.
*This feature is under development.*

### Rx (RF Receive)
The Rx page allows one to visualize time-domain MR signal, generated from an arbitrary grayscale image, and see the effects of using different demodulation frequencies and ADC sampling rate.
*This feature is under development.*

### PSD Inspector
The Pulse Sequence Diagram (PSD) Inspector allows you to load MR pulse sequences in the pulseq .seq format and visualize them.  
*This feature is under development.*

### Phantom Viewer
*This feature is under development.*

### Reconstruction
The recon page allows experimenting with different reconstruction methods, including Cartesian, non-Cartesian, and deep learning (link to DRUNK) options.
*This feature is under development.*

## References

Kose, R., & Kose, K. (2017). BlochSolver: a GPU-optimized fast 3D MRI simulator for experimentally compatible pulse sequences. Journal of Magnetic Resonance, 281, 51-65.

Layton, K. J., Kroboth, S., Jia, F., Littin, S., Yu, H., Leupold, J., ... & Zaitsev, M. (2017). Pulseq: a rapid and hardware-independent pulse sequence prototyping framework. Magnetic resonance in medicine, 77(4), 1544-1552.
