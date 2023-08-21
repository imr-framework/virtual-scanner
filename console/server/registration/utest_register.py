# Copyright of the Board of Trustees of Columbia University in the City of New York

import unittest
import virtualscanner.server.registration.register as reg

"""
#. This script unit tests the subject registration
#. tests multiple cases of the registration implementation

Returns
-------
    status : int
        | 0: tests passed
        | 1: fail
"""


class MyTestCase(unittest.TestCase):

    def create_payload(self):
        """
        This definition creates a test payload.

        Returns
        -------
        payload : dict
            All fields required by the REGISTRATION table in subject.db
        """
        # can make this more random later on, but for now we hardcode
        payload = {
            "SUBJECTTYPE": "numerical",
            "patid": 5466,  # random.randint(1, 1000)
            "name": "Numerical",
            "AGE": 0,  # random.randint(4, 100),
            "DOB": "4/17/2019",  # redundant but useful for an OR situation later
            "GENDER": "other",
            "WEIGHT": 3.0,  # random.randint(60, 300),
            "HEIGHT": 20.0,  # random.randint(60, 200),
            "ORIENTATION": "HFS",
            "ANATOMY": "brain"

        }
        status = reg.consume(payload)

        # check for existing subject
        payload = {
            "PATID": 5465,
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
        print(payload)


if __name__ == '__main__':
    unittest.main()
