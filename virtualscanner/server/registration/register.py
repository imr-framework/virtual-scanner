"""
This script registers a subject
Parameters
----------
    payload - type dict

Performs
--------
    Creates a database if it does not exist
    Appends registration data


Returns
-------
    status - type int

Unit Test app
-------------
     utest_register
Author: Sairam Geethanath
Date: 03/06/2019
Version 0.0
Copyright of the Board of Trustees of  Columbia University in the City of New York
"""

import os

import virtualscanner.server.registration.db_operations_mgr as dbom

db_path = os.path.join("./src/server/registration/", "subject.db")


def consume(payload):
    """
    Parameters
    ----------
    payload - type dict

    Returns
    -------
    status - type int
    payload - type dict; in case of reusing an existing registered subject
    """

    # Check if  database 'subject' and table 'registration' exist, if not create them
    if (not (os.path.isfile(db_path))):
        dbom.create()

    # Add current registration payload to subject/registration and return status
    status = dbom.insert(payload)
    print(status)
    return (status)


# def check(payload):

def reuse(payload):
    rows = dbom.query(payload)
    return rows
