<p align="center"> <a>
<img title="Virtual Scanner Logo" src="https://github.com/imr-framework/imr-framework.github.io/blob/master/img/portfolio/virtual-scanner.png" width="225">
</a></p>
<h1 align="center"> Virtual Scanner </h1> <br>

Virtual Scanner is an end-to-end hybrid Magnetic Resonance Imaging (MRI) simulator/console designed to be zero-footprint, modular, and supported by open-source standards.

The project is a response to the [ISMRM 2019 Junior Fellow Challenge (Africa)](https://www.ismrm.org/2019-junior-fellow-challenge/africa/), which poses the task of boosting accessibility to MRI training resources for underserved areas such as sub-Saharan Africa. We designed Virtual Scanner to help develop local expertise in these areas so that sustained deployment of MRI hardware is possible. Importantly, Virtual Scanner will be continually developed as a research tool that provides functionalities for simulating and prototyping MRI acquisition methods as well as services for sharing computational methods and resources with researchers around the world.

Virtual Scanner consists of two modes: in Standard Mode, a console-like GUI allows users to perform virtual scans and conduct basic analysis; in Advanced Mode, modular simulation/analysis of the entire signal chain may be performed.  

## Quick Start
1. Install Python 3.6.x.
2. Create and activate a `venv` virtual environment.
3. In a terminal: `pip install virtual-scanner`.

If you want to hack around with the code:
1. Install Python 3.6.x.
2. Clone the repository and `cd` into it.
3. Create and activate a `venv` virtual environment.
4. Install dependencies by running `pip install -r requirements.txt` in a terminal.

Then, run [`coms_server_flask.py`](https://github.com/imr-framework/virtual-scanner/blob/ISMRM2019/virtualscanner/coms/coms_ui/coms_server_flask.py) to start the browser GUI.

To access the browser app, there are two ways:
* Local hosting (only on the laptop running the script): go to the generated link (http://0.0.0.0:5000/) if you are a mac user, and this link (http://127.0.0.1:5000) if you are a windows user.  
* Remote hosting : look up your IP address. Suppose it's 123.45.67.890, then you can go to (http://123.45.67.890:5000) to connect to the server remotely, either on the serving machine or different machines on the same network.

Now you can start playing with Virtual Scanner! Log in with your email address, select Standard or Advanced mode, and click "Begin Scan". Instructions for each tab are given in the [Wiki](https://github.com/imr-framework/virtual-scanner/wiki).

Alternatively, you can read the API documentation [here](https://imr-framework.github.io/virtual-scanner/) and run the Python test scripts in each module with more options available than allowed on the GUI.

## Standard Mode
* The **Register** page allows you to choose a phantom for simulation. Its format is similar to the form for entering information of the subject when conducting real scans. Choose the "Numerical" phantom for all simulations now.

* The **Acquire** page allows the user to choose either a Gradient Echo (GRE) or a Spin Echo (SE, with optional inversion recovery) sequence, enter the parameters, and simulate them on a cylindrical phantom ("Numerical") containing spheres with different T1, T2, and PD values.

* The **Analyze** page allows the user to load a series of data acquired in ISMRM/NIST phantom for T1 or T2 mapping and conduct curve fitting to obtain T1 and T2 maps. In addition, it can detect spheres in the phantom, a feature useful for comparing generated parameter values to literature values.

## Advanced Mode
* The **Tx** (RF transmit) page allows one to calculate and plot SAR from pulseq .seq files. *This feature is under development.*

* The **Rx** (RF receive) page allows one to visualize time-domain MR signal, generated from an arbitrary grayscale image, and see the effects of using different demodulation frequencies and ADC sampling rate. *This feature is under development.*

* Other features, including phantom and sequence viewers and reconstruction methods, are in active development.

## Known Issues
Please refer to the [Known Issues](https://github.com/imr-framework/virtual-scanner/blob/master/KNOWN_ISSUES.md) document.

## Contributing
If you would like to contribute to Virtual Scanner, please take a look at the [Community Guidelines](https://github.com/imr-framework/virtual-scanner/blob/master/CONTRIBUTING.md) document.
