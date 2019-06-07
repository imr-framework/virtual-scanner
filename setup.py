import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='virtual-scanner',
    version='1.0.21',
    author='imr-framework',
    author_email='imr.framework2018@gmail.com',
    description='Virtual Scanner educational tool for MRI',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/imr-framework/Virtual-Scanner',
    packages=setuptools.find_packages(),
    install_requires=['absl-py==0.7.1',
                      'astor==0.7.1',
                      'Click==7.0',
                      'cycler==0.10.0',
                      'Flask==1.0.2',
                      'gast==0.2.2',
                      'grpcio==1.20.0',
                      'h5py==2.9.0',
                      'itsdangerous==1.1.0',
                      'Jinja2==2.10.1',
                      'Keras==2.2.4',
                      'Keras-Applications==1.0.7',
                      'Keras-Preprocessing==1.0.9',
                      'kiwisolver==1.0.1',
                      'Markdown==3.1',
                      'MarkupSafe==1.1.1',
                      'matplotlib==3.0.3',
                      'mock==2.0.0',
                      'nibabel==2.4.0',
                      'numpy==1.16.2',
                      'opencv-python==4.0.0.21',
                      'pbr==5.1.3',
                      'Pillow==6.0.0',
                      'protobuf==3.7.1',
                      'pydicom==1.2.2',
                      'pyparsing==2.3.1',
                      'pypulseq==0.0.3',
                      'python-dateutil==2.8.0',
                      'PyYAML==5.1',
                      'scipy==1.2.1',
                      'six==1.12.0',
                      'tensorboard==1.13.1',
                      'tensorflow==1.13.1',
                      'tensorflow-estimator==1.13.0',
                      'termcolor==1.1.0',
                      'Werkzeug==0.15.2'],
    license='License :: OSI Approved :: GNU Affero General Public License v3',
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'virtual-scanner = virtualscanner.coms.coms_ui.coms_server_flask:launch_virtualscanner'
        ]
    },
)
