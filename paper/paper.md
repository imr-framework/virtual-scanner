---
title: 'Virtual Scanner: MRI on a Browser'

tags:
  - Accessible Magnetic Resonance Imaging
  - Web Application
  - Python
  - Pulseq

authors:
  - name: Gehua Tong
    orcid: 0000-0001-6263-762X
    affiliation: 1
  - name: Sairam Geethanath
    orcid: 0000-0002-3776-4114
    affiliation: 1
  - name: Marina Jimeno Manso
    orcid: 0000-0002-1141-2049
    affiliation: 1
  - name: Enlin Qian
    orcid: 0000-0001-7531-1274
    affiliation: 1
  - name: Keerthi Sravan Ravi
    orcid: 0000-0001-6886-0101
    affiliation: 1
  - name: Nishika Girish
    affiliation: 2
  - name: John Thomas Vaughan Jr.
    orcid: 0000-0002-6933-3757
    affiliation: 1

affiliations:
 - name: Columbia Magnetic Resonance Research Center, Columbia University in the City of New York
   index: 1
 - name: Plum Grove Junior High, Palatine, IL, USA
   index: 2
 
date: 26 July 2019

bibliography: paper.bib
---
# Summary
Magnetic Resonance Imaging (MRI) is a medical imaging modality that provides excellent soft-tissue contrast with high diagnostic value. 
System simulators are key to method development in MRI because of its inherent complexity. However, most existing MR system-wide simulators are proprietary to manufacturing companies, and an open-source, integrated webtool would help MR researchers around the world develop and share new methodology. Specifically, it would provide vendor-neutral tools for rapidly prototyping MR acquisition methods and hardware components, services for sharing computational methods and resources, and education for under-served areas to develop their own MR expertise [@obungoloch_2019]. Being vendor-neutral would also enable multi-site repeatability studies. To fulfill this need, we designed ``Virtual Scanner`` with three main characteristics in mind:


* Modular: ``Virtual Scanner`` consists of modules corresponding to steps in the MR signal chain [@webb2016magnetic]. Since modern MRI systems contain multiple levels of hardware and software, it is important for researchers to select the appropriate steps for simulation when developing acquisition paradigms, analysis methods, or hardware components. Students may also choose to focus on individual aspects of the experiment.

* Zero-footprint: ``Virtual Scanner`` was implemented as a web application using the FLASK framework [@flask] with backend code in Python. The local server can be set up by running a script or using the command line. Once the server is up, one can access it on a browser without additional steps for installing any software packages. 

* Open-source: existing tools in the MR community are incorporated for effective creation and sharing of resources. Pulseq, a multi-vendor MR pulse sequence format [@layton2017pulseq; @ravi2018pulseq], can be directly fed into the simulator. Since translating from simulation to experiments on a real scanner is a key step in MR methods development, being able to deploy the same sequence file can help streamline this process. 

## Standard Mode
The Standard Mode mimics a real scanner console and enables the user to perform custom MRI experiments. This mode may be used by MR technicians as practice for scanner console operation without requiring expensive scanner hours and can help increase access to education [@geethanath2019accessible]. 

The virtual experiments are simulated with a discrete event approach using the Bloch equation [@kose2017blochsolver]. The backend code may be employed by researchers with custom sequences and phantoms that are straightforward to set up and test.  

The Graphical User Interface (GUI) consists of three pages:

* Register: similar to patient registration, this page currently allows selection of a numerical phantom. Details such as weight and age are included for MR safety evaluation and future inclusion of human anatomy. 

* Acquire: this page allows setting parameters for standard MRI sequences such as Spin Echo (SE), Gradient Recalled Echo (GRE), and Inversion Recovery Spin Echo (IRSE) and applying them on the phantom to obtain MR images. A pulse sequence library script, powered by a Python implementation of Pulseq [@ravi2018pulseq], custom-generates the sequences as standardized objects and files. 

* Analyze: this page obtains relaxation time (T1 and T2) maps from series of images with different acquisition parameters using curve-fitting [@brown2014magnetic]. T1 and T2 are two main sources of contrast in anatomical MR images, and their quantitative estimation can help distinguish finer differences between tissue types.

## Advanced Mode
The Advanced Mode is envisioned to be a system-wide MR simulator. At this moment, RF Transmission (Tx) and RF Reception (Rx) have been implemented.

* Tx: Specific Absorption Rate (SAR) over time is calculated directly from pulse sequences [@Graesslin_SAR] in the Pulseq format [@ravi2018pulseq]. This measure ensures patient safety from RF heating, and is a useful check for custom sequences. 

* Rx: time-domain MR signals are generated from the spatial frequency domain of grayscale images and put through demodulation, ADC sampling, and reconstruction steps that consist the MR receive chain. 

## Roadmap 

More features are undergoing active development for both modes. For Standard Mode, a human brain phantom will be added, as well as additional pulse sequences including Echo Planar Imaging (EPI) [@mansfield1977multi; @stehling1991echo] and Magnetization Prepared RApid Gradient Echo (MPRAGE) [@mugler1990three], oblique spatial encoding, and interactive plotting of parameter maps. For Advanced Mode, a Pulse Sequence Diagram (PSD) viewer, RF modeling, and B0 map will be included. Furthermore, we plan to accelerate the simulator, include more physical effects, and unlock advanced reconstruction methods by incorporating existing open source tools such as JEMRIS [@stocker2010high], GPI [@zwart2015graphical], and BART [@uecker2015berkeley].

## Projected Usage 

* Tool: ``Virtual Scanner`` is expected to boost the efficiency of developing novel MRI acquisition methods. We have found it helpful for prototyping and checking the validity of Pulseq sequences and plan to use it as the simulator for an in-house project of developing an accessible MRI scanner. 

* Service: in the near future, we aim to provide ``Virtual Scanner`` as a free online service for sharing computational methods and resources to the MR community. Examples include advanced iterative or machine learning based image reconstruction, SAR calculation, and custom sequence simulation.

* Education: we plan to deploy ``Virtual Scanner`` for an MRI course at Mbarara University in Uganda starting from September 2019 for disseminating MR knowledge.

# Acknowledgements
``Virtual Scanner`` was funded in part by the Seed Grant Program for MR Studies of the Zuckerman Mind Brain Behavior Institute at Columbia University (PI: Geethanath), and was developed at Zuckerman Mind Brain Behavior Institute MRI Platform, a shared resource.

# References




