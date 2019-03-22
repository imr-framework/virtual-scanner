"""
This script runs the server based on the Flask package and hosts the GUIs on the web
Parameters
----------
    payload,default = None

Returns
-------
    payload or status on ping from clients

Performs
--------
   Tx to client
   Rx from client

Unit Test app
-------------
     utest_coms_flask
Author: Sairam Geethanath , Modified by: Marina Manso Jimeno
Date: 03/22/2019
Version 0.0
Copyright of the Board of Trustees of  Columbia University in the City of New York
"""
import os
import sys
from flask import Flask, render_template, request, redirect, Response
import random, json

# Define the location of template and static folders
template_dir = os.path.abspath('../templates')
static_dir=os.path.abspath('../static')

app = Flask(__name__,template_folder=template_dir,static_folder=static_dir)


@app.route('/')  # This needs to point to the login screen and then we can use the register link seprately
def on_register():
    """

         Parameters
        ----------
           void

        Returns
        -------
           void (status in debug mode if required)

        Performs
        --------
            Renders the registration html page on the web
    """

    # serve register template
    return render_template('register.html')


@app.route('/acquire.html')
def on_acq():
    """

        Parameters
        ----------
               void

         Returns
        -------
           void (status in debug mode if required)

        Performs
        --------
            Renders the acquisition html page on the web
        """
    # serve index template
    return render_template('acquire.html')


@app.route('/receiver', methods=['POST'])
def worker():
    """

            Parameters
            ----------
                   void

             Returns
            -------
               payload

            Performs
            --------
                Rx payload from the client
                TODO: invokes payload
            """
    # read json + reply
    payload = request.data
    # data = request.get_json() #Needs debugging, interesting TODO:
    print(payload)
    result = ''
    return result


if __name__ == '__main__':
    # run!
    # app.run(host='0.0.0.0', debug=True)
    app.run(host='0.0.0.0', debug=True)
