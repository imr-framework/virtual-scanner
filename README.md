<p align="center"> <a>
    <img title="Virtual Scanner Logo" src="https://github.com/imr-framework/imr-framework.github.io/blob/master/img/portfolio/virtual-scanner.png" width="225">
  </a></p>
<h1 align="center"> Virtual Scanner </h1> <br>

Virtual Scanner is an end-to-end hybrid MR simulator/console designed to be easily accessible, modular, and supported by open-source standards. 

The project is a response to the [ISMRM 2019 Junior Fellow Challenge (Africa)](https://www.ismrm.org/2019-junior-fellow-challenge/africa/).

Virtual Scanner consists of two modes: in Standard Mode, a console-like GUI allows users to perform virtual scans and conduct basic analysis; in Advanced Mode, modular simulation/analysis of the entire MR signal chain may be performed.  

## Quick Start
First, clone the repository. Make sure you have all packages listed in requirements.txt installed in your Virtual Environment.

Then, run coms_server_flask.py to start the browser GUI and follow the generated link to access it. Instructions for each tab are given in the [Wiki](https://github.com/imr-framework/virtual-scanner/wiki).

Alternatively, you can read the API documentation (link) and run the Python test scripts in each module with more options available than allowed on the GUI.


## Standard Mode
* The **Register** page allows you to choose a phantom for simulation. Its format is similar to the form for entering information of the subject when conducting real scans. Choose the "Numerical" phantom for all simulations now. 

* The **Acquire** page allows the user to choose either a Gradient Echo (GRE) or a Spin Echo (SE, with optional inversion recovery) sequence, enter the parameters, and simulate them on a cylindrical phantom ("Numerical") containing spheres with different T1, T2, and PD values. 

* The **Analyze** page allows the user to load a series of data acquired for T1 or T2 mapping and conduct curve fitting to obtain T1 and T2 maps. In addition, it can detect circles on the maps, a common feature of real MR phantoms.

## Advanced Mode
* The **Tx** (RF transmit) page allows one to calculate and plot SAR from pulseq .seq files.
*This feature is under development.*

* The **Rx** (RF receive) page allows one to visualize time-domain MR signal, generated from an arbitrary grayscale image, and see the effects of using different demodulation frequencies and ADC sampling rate. 
*This feature is under development.*

* Other features, including phantom and sequence viewers and reconstruction methods, are in active development. 

## References

Kose, R., & Kose, K. (2017). BlochSolver: a GPU-optimized fast 3D MRI simulator for experimentally compatible pulse sequences. Journal of Magnetic Resonance, 281, 51-65.

Layton, K. J., Kroboth, S., Jia, F., Littin, S., Yu, H., Leupold, J., ... & Zaitsev, M. (2017). Pulseq: a rapid and hardware-independent pulse sequence prototyping framework. Magnetic resonance in medicine, 77(4), 1544-1552.
