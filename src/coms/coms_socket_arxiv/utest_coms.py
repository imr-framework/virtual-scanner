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
import utest_register as ureg
import coms_sender



def create_payload(coms_dir):
    # can make this more random later on, but for now we hardcode
    data = ureg.create_payload()
    if coms_dir == "Tx":
        payload = {
            "action": "Tx",
            "data": data
        }
    elif coms_dir == "Rx":
        payload = {
            "action": "Rx",
            "data": data
        }
    return payload


payload = create_payload("Tx")
coms_sender.exec(payload)


# coms_server.coms_server_exec(newthread)
