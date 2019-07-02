# Copyright of the Board of Trustees of  Columbia University in the City of New York
"""
This script unit starts and tests the communications between server and client(s).
Requires `coms_server_flask` to be running before the unit test is run (run `coms_server_flask` first).
"""

import webbrowser

# This URL needs to be changed to the server address if running on a remote client on the local network
webbrowser.open("http://0.0.0.0:5000/")
