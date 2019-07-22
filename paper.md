---
title: 'Virtual Scanner: MRI on a Browser'

tags:
  - Magnetic Resonance Imaging
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
  - name: John Thomas Vaughan
    orcid: 0000-0002-6933-3757
    affiliation: 1

affiliations:
 - name: Columbia Magnetic Resonance Research Center, Columbia University in the City of New York
   index: 1
 - name: Plum Grove Junior High, Palatine, IL, USA
   index: 2
 
date: 22 July 2019

bibliography: paper.bib
---
# Summary
Magnetic Resonance Imaging (MRI) is a medical imaging modality that provides excellent soft-tissue contrast with high diagnostic value. Well-trained personnel are required for sustaining deployment of MRI systems because of their inherent complexities. In underserved or remote areas, local expertise is insufficiently established, which formed the motivation for this design challenge [@obungoloch_2019]. 

In response, we built ``Virtual Scanner``[@vsrepo], an MR system simulator employable for both educational and research purposes. On one hand, a free and comprehensive MR education tool can help maximize the long-term influence of hardware in underserved areas. On the other hand, most existing MR system simulators are proprietary to manufacturing companies and an open-source, integrated webtool will help MR researchers around the world develop and share new methodology.

``Virtual Scanner``[@vsrepo] has three main characteristics:

* Modular: modern MRI systems are equipped with multiple coils for producing magnetic fields, radiofrequency (RF) electronics, console software, and related accessories. ``Virtual Scanner`` consists of modules corresponding to steps in the signal chain [@webb2016magnetic]. Trainees can learn about the imaging experiment by choosing modules that cater to their practical needs.

* Zero-footprint: ``Virtual Scanner`` was implemented as a web application using the FLASK framework [@flask] with backend code in Python. The local server can be set up by running a script or using the command line. Once the server is up, one can access it on a browser without additional steps for installing any software packages. 

* Open-source: existing tools in the MR community are incorporated for effective creation and sharing of resources. Pulseq, a multi-vendor MR pulse sequence format [@layton2017pulseq; @ravi2018pulseq], can be directly fed into the simulator. This is especially helpful when users plan to translate from simulation to experiments in a real scanner, since they may use the same sequence file for both purposes.

## Standard Mode
The Standard Mode mimics a real scanner console and enables the user to perform custom MRI experiments. These virtual experiments are simulated with an event-approach using the Bloch equation [@kose2017blochsolver]. 

This mode is intended for MR technicians and allows practice for scanner console operation without requiring expensive scanner hours. In this way, more technicians can be trained per unit time in places with few scanners per capita [@geethanath2019accessible]. Although practice on the real scanner is still necessary, this provides a good introduction to setting up MRI scans. The Graphical User Interface (GUI) consists of three pages:

* Register: similar to patient registration, this page currently allows the trainee to select a numerical phantom. Details such as weight and age are included for MR safety evaluation and future inclusion of human anatomy. 

* Acquire: this page allows setting parameters for standard MRI sequences such as Spin Echo (SE), Gradient Recalled Echo (GRE), and Inversion Recovery Spin Echo (IRSE) and applying them on the phantom to obtain MR images. A pulse sequence library script, powered by a Python implementation of Pulseq [@ravi2018pulseq], custom-generates the sequences as standardized .seq objects and files. 

* Analyze: this page obtains relaxation time (T1 and T2) maps from series of images with different acquisition parameters using curve-fitting [@brown2014magnetic]. T1 and T2 are two main sources of contrast in anatomical MR images, and their quantitative estimation can help distinguish finer differences between pathological and normal tissue. 

## Advanced Mode
The Advanced Mode is envisioned to be an MR system simulator that can be used flexibly for different aspects of MR education. At this moment, RF Transmission (Tx) and RF Reception (Rx) have been implemented.

* Tx: Specific Absorption Rate (SAR) over time is calculated directly from pulse sequences [@Graesslin_SAR] in the Pulseq format [@ravi2018pulseq]. This measure ensures patient safety from RF heating, and is useful for evaluating custom sequences for practice. 

* Rx: time-domain MR signals are generated from the spatial frequency domain of grayscale images and put through demodulation, ADC sampling, and reconstruction steps that consist the MR receive chain.

## Roadmap 

More features are undergoing active development for both modes. For Standard Mode, a human brain phantom will be added, as well as more pulse sequences including Echo Planar Imaging (EPI) [@mansfield1977multi;@stehling1991echo] and Magnetization Prepared RApid Gradient Echo (MPRAGE) [@mugler1990three], oblique spatial encoding, and interactive plotting of parameter maps. For Advanced Mode, a Pulse Sequence Diagram (PSD) viewer, RF modeling, and B0 map will be included. Furthermore, we plan to accelerate the simulator, include more physical effects, and unlock advanced reconstruction methods by incorporating existing open source tools such as JEMRIS [@stocker2010high], GPI [@zwart2015graphical], and BART [@uecker2015berkeley].

## Projected Usage 

* Tool: ``Virtual Scanner`` is expected to boost the efficiency of developing novel MRI acquisition methods. We have found it helpful for prototyping and checking the validity of Pulseq sequences before using it on a scanner.

* Service: in the near future, we aim to provide ``Virtual Scanner`` as a free online service for sharing computational methods and resources to the MR community. Examples include advanced iterative or machine learning based image reconstruction, SAR calculation, and custom sequence simulation.

* Education: we plan to deploy ``Virtual Scanner`` for an MRI course at Mbarara University in Uganda starting from September 2019 for disseminating MR knowledge.

# Acknowledgements
``Virtual Scanner`` was funded in part by the Seed Grant Program for MR Studies of the Zuckerman Mind Brain Behavior Institute at Columbia University (PI: Geethanath), and was developed at Zuckerman Mind Brain Behavior Institute MRI Platform, a shared resource.

# References


