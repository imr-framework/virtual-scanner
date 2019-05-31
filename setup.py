from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='virtual-scanner-imr-framework',
    version='0.0.0',
    author='imr-framework',
    author_email='imr.framework2018@gmail.com',
    description='Virtual Scanner educational tool for MRI',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/imr-framework/Virtual-Scanner',
    packages=['virtualscanner', 'virtualscanner.coms', 'virtualscanner.coms.coms_ui',
              'virtualscanner.coms.coms_ui.static', 'virtualscanner.coms.coms_ui.static.recon',
              'virtualscanner.coms.coms_socket_arxiv', 'virtualscanner.server', 'virtualscanner.server.rf',
              'virtualscanner.server.rf.tx', 'virtualscanner.server.rf.tx.SAR_calc', 'virtualscanner.server.rx',
              'virtualscanner.server.ana', 'virtualscanner.server.recon', 'virtualscanner.server.recon.drunck',
              'virtualscanner.server.simulation', 'virtualscanner.server.simulation.bloch',
              'virtualscanner.server.registration', 'virtualscanner.server.quant_analysis', 'virtualscanner.pypulseq2',
              'virtualscanner.pypulseq2.utils', 'virtualscanner.pypulseq2.examples',
              'virtualscanner.pypulseq2.examples.old', 'virtualscanner.pypulseq2.Sequence'],
    license='License :: OSI Approved :: GNU Affero General Public License v3'
)
