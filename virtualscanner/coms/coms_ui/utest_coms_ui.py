"""
This script unit starts and tests the communications between server and client(s)
Parameters
----------
    void
    Requires coms_server_flask to be running before the unit test is run (i.e.: run coms_server_flask.py first)

Returns
-------
    payload


Performs
--------
    tests multiple cases of the server client interactions via the html pages rendered


Author: Sairam Geethanath
Date: 03/11/2019
Version 0.0
Copyright of the Board of Trustees of  Columbia University in the City of New York
"""

import webbrowser
webbrowser.open("http://0.0.0.0:5000/") #This URL needs to be changed to the server address if running on a remote client on the local network




