# Contributing to Virtual-Scanner
:thumbsup: :tada: First off, thanks for taking time to contribute! :thumbsup: :tada:

The following is a set of guidelines for contributing to Virtual-Scanner. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## Table of contents
1. [Code of Conduct](#code-of-conduct)
2. [PEP Style Guide for Python coding](#style-guide-for-python-code)
3. [Source code header](#source-code-header)

## Code of Conduct
This project and everyone participating in it is governed by the [Virtual-Scanner Code of Conduct](https://github.com/imr-framework/Virtual-Scanner/blob/ISMRM2019/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to imr.framework2018@github.com.

## Style Guide for Python Code
Read through the [Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/) and follow the coding conventions. If you notice any of Virtual-Scanner's code not adhering to PEP8, submit a pull request or open an issue.

## Source code header
Insert and fill out the following header comments at the top of every script in your source code:
```
Institution : (your university/research facility)
Version : 1.0.0 
```
Note that "Version" refers to the Virtual Scanner release you developed your code on. You can add more fields if needed. 

## Documenting source code
Please add a top-level description of code functionality in each script. In addition, document every class and method in your source code using the [Numpy docstring guide](https://numpydoc.readthedocs.io/en/latest/format.html). An example is shown below.

```python
def addition(a, b):
  """
  Addition of two numbers.
  
  Parameters
  ----------
  a : int, float
    Operand 1
  b : int, float
    Operand 2
    
  Returns
  -------
  Arthimetic sum of operand 1 and operand 2.
  """
  return a + b
```
