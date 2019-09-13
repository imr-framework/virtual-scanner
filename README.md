<p align="center"> <a>
<img title="Virtual Scanner Logo" src="https://github.com/imr-framework/imr-framework.github.io/blob/master/img/portfolio/virtual-scanner.png" width="225">
</a></p>
<h1 align="center"> Virtual Scanner </h1> <br>

Virtual Scanner is an end-to-end hybrid Magnetic Resonance Imaging (MRI) simulator/console designed to be zero-footprint, modular, and supported by open-source standards.

The project is a response to the [ISMRM 2019 Junior Fellow Challenge (Africa)](https://www.ismrm.org/2019-junior-fellow-challenge/africa/), which poses the task of boosting accessibility to MRI training resources for underserved areas such as sub-Saharan Africa. We designed Virtual Scanner to help develop local expertise in these areas so that sustained deployment of MRI hardware is possible. Importantly, Virtual Scanner will be continually developed as a research tool that provides functionalities for simulating and prototyping MRI acquisition methods as well as services for sharing computational methods and resources with researchers around the world.

Virtual Scanner consists of two modes: in Standard Mode, a console-like GUI allows users to perform virtual scans and conduct basic analysis; in Advanced Mode, modular simulation/analysis of the entire signal chain may be performed.  

## Quick Start
If you just want to get started with using Virtual Scanner:
1. Install Python 3.6.x.
2. Create and activate a virtual environment of your choice. For example, using `virtualenv`, it would be: `python -m venv virtualscanner_env`
3. In your terminal: `pip install virtual-scanner`, and finally
4. `virtualscanner`

The browser application should have started running. To access the browser app, there are two ways:
* Local hosting (only on the laptop running the script): go to the generated link (http://0.0.0.0:5000/) if you are a mac user, and this link (http://127.0.0.1:5000) if you are a windows user.  
* Remote hosting : look up your IP address. Suppose it's 123.45.67.890, then you can go to (http://123.45.67.890:5000) to connect to the server remotely, either on the serving machine or different machines on the same network.

Now you can start playing with Virtual Scanner! Log in with your email address, select Standard or Advanced mode, and click "Begin Scan". Instructions for each tab are given in the [Wiki](https://github.com/imr-framework/virtual-scanner/wiki).

## Docker Start
This would be a good solution if you don't want to install the large number of dependencies to your host. We have tested Docker on Windows and Mac.

If you use a Windows 10 Pro system, first download Docker desktop and then run the following in the command line:

```bash
$ docker run -p 5000:5000 imrframework/virtual-scanner:latest
```

This downloads the pre-built image from Docker. You can then open up your browser to [127.0.0.1:5000](127.0.0.1:5000) to see the interface. 

If you use a Windows 10 Home system, please use an alternate approach (clone the repository or pip install) because Docker isn't supported on this system and we have not succeeded in using the image with Docker Toolbox. 

If you use a Mac, you can either download the image as described above or you can locally build a Docker container to run the application in the following way:

First, build the container from the [Dockerfile](Dockerfile):

```bash
$ docker build -t virtualscanner .
```

Then you can run the container and bind port 5000 to expose the application to the host:

```bash
$ docker run -p 5000:5000 virtualscanner
```

And open up your browser to [0.0.0.0:5000](0.0.0.0:5000)to see the interface.
The container is **not intended** for a production deployment, but rather is appropriate for local usage.


## Pro Start
If you want to hack around with the code:
1. Install Python 3.6.x.
2. Create and activate a virtual environment of your choice. Here's a [tutorial][pycharm-venv] if you are using the 
PyCharm IDE, for example.
3. Clone the repo and `cd` into it.
4. `pip install -e .`.
5. Open the repo in your favourite IDE, hack around with the code.
6. Run `virtualscanner/coms/coms_ui/coms_server_flask.py` to run your changes.

Read the API documentation [here](https://imr-framework.github.io/virtual-scanner/) and run the Python test scripts in each module with more options available than allowed on the GUI.

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

[pycharm-venv]: https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html
