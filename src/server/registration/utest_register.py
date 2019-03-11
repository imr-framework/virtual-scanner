"""
This script unit tests the subject registration
Parameters
----------
    void

Performs
--------
    tests multiple cases of the registration implementation


Returns
-------
    status - type int

Author: Sairam Geethanath
Date: 03/07/2019
Version 0.0
Copyright of the Board of Trustees of  Columbia University in the City of New York
"""
import register as reg
import random
import os


# os.chdir("./Virtual-Scanner")

def create_payload():
    # can make this more random later on, but for now we hardcode
    payload = {
        "ID": random.randint(1, 1000),
        "NAME": "Mustang",
        "AGE": random.randint(4, 100),
        "DOB": "03/03/1983",  # redundant but useful for an OR situation later
        "GENDER": "Female",
        "WEIGHT": random.randint(60, 300),
        "HEIGHT": random.randint(60, 200),
        "ORIENTATION": "Head first supine",
        "ANATOMY": "Brain"

    }
    return payload


# Simulate the payload to be received from the client
payload = create_payload()

# check for registering a subject
status = reg.consume(payload)
print(status)

# check for existing subject
payload = {
    "ID": 205,
    # "NAME": "Mustang",
    # "AGE": 45,
    #  "DOB": "03/03/1983", #redundant but useful for an OR situation later
    #  "GENDER": "Female",
    #  "WEIGHT": 170,
    #  "HEIGHT": 165,
    #  "ORIENTATION":"Head first supine",
    # "ANATOMY":"Brain"

}
payload = reg.reuse(payload)

# TODO: handle different cases