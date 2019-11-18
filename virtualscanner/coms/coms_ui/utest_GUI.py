# Copyright of the Board of Trustees of  Columbia University in the City of New York
"""
This script unit starts and tests the communications between server and client(s).
"""

import threading
from virtualscanner.coms.coms_ui import GUI_test_functions
from virtualscanner.coms.coms_ui import coms_server_flask
import time
# import webbrowser
# import subprocess
# from virtualscanner.utils import constants


def selenium_function():
    time.sleep(10)
    # This URL needs to be changed to the server address if running on a remote client on the local network
    # webbrowser.open("http://0.0.0.0:5000/")
    GUI_test_functions.launch_tests()
    # Do all tests as you want here and get 200 responses.
    # Figure out how to report 200 responses if required
    # webbrowser.close()


if __name__ == '__main__':
    t = threading.Thread(target=selenium_function)
    t.daemon = True
    t.start()
    coms_server_flask.launch_virtualscanner()
