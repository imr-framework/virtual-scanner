"""
This script unit tests the communications between server and client(s)
Parameters
----------
    void
    Requires coms_listener to be running before the unit test is run

Performs
--------
    tests multiple cases of the server client interactions


    Returns
-------
    status - type int

Author: Sairam Geethanath
Date: 03/11/2019
Version 0.0
Copyright of the Board of Trustees of  Columbia University in the City of New York
"""
if __name__ == '__main__':
    import os
    import sys

    script_path = os.path.abspath(__file__)
    SEARCH_PATH = script_path[:script_path.index('Virtual-Scanner') + len('Virtual-Scanner') + 1]
    sys.path.insert(0, SEARCH_PATH)
import virtualscanner.server.registration.utest_register as ureg


def create_payload(coms_dir):
    # can make this more random later on, but for now we hardcode
    data = ureg.create_payload()
    if coms_dir == "tx":
        payload = {
            "action": "tx",
            "data": data
        }
    elif coms_dir == "rx":
        payload = {
            "action": "rx",
            "data": data
        }
    return payload


payload = create_payload("tx")
coms_sender.exec(payload)

# coms_server.coms_server_exec(newthread)
