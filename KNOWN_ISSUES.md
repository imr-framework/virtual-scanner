# Known Issues

## [Matplotlib issue on Mac when using (Ana)conda](issue26)
#### Platform:
(Ana)conda/macOS

#### Steps to reproduce:
1. `python virtualscanner/coms/coms_ui/coms_server_flask.py`

#### Expected behaviour:
The Virtual Scanner browser app is functional on the browser at the localhost
address.

#### Observed behaviour:
> ImportError: Python is not installed as a framework. The Mac OS X backend will not be able to function correctly if Python is not installed as a framework. See the Python documentation for more information on installing Python as a framework on Mac OS X. Please either reinstall Python as a framework, or try one of the other backends. If you are using (Ana)Conda please install python.app and replace the use of 'python' with 'pythonw'. See 'Working with Matplotlib on OSX' in the Matplotlib FAQ for more information.

#### Fix:
1. `echo "backend: TkAgg" >> ~/.matplotlib/matplotlibrc`

#### Notes:
https://github.com/scikit-optimize/scikit-optimize/issues/637#issuecomment-369730420

[issue26]: https://github.com/imr-framework/virtual-scanner/issues/26
