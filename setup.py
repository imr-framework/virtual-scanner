from pathlib import Path

import setuptools

here = Path(__file__).parent

with open(str(here / 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open(str(here / 'requirements.txt'), 'r') as f:
    install_reqs = f.read().strip()
    install_reqs = install_reqs.split("\n")

setuptools.setup(
    name='virtual-scanner',
    version='1.0.24',
    author='imr-framework',
    author_email='imr.framework2018@gmail.com',
    description='Virtual Scanner educational tool for MRI',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/imr-framework/virtual-scanner',
    packages=setuptools.find_packages(),
    install_requires=install_reqs,
    license='License :: OSI Approved :: GNU Affero General Public License v3',
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'virtualscanner = virtualscanner.coms.coms_ui.coms_server_flask:launch_virtualscanner'
        ]
    },
)
