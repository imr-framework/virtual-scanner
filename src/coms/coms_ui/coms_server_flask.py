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

if __name__ == '__main__':
    import os
    import sys

    script_path = os.path.abspath(__file__)
    SEARCH_PATH = script_path[:script_path.index('Virtual-Scanner') + len('Virtual-Scanner') + 1]
    sys.path.insert(0, SEARCH_PATH)

import json

from flask import Flask, render_template, request, redirect, session

import src.server.registration.register as reg
from src.server.simulation.bloch import caller_script_blochsim as bsim

# Define the location of template and static folders
# template_dir = os.path.abspath('../templates')
# static_dir=os.path.abspath('../static')

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

users = []
app.secret_key = 'Session_key'


@app.route('/', methods=['POST',
                         'GET'])  # This needs to point to the login screen and then we can use the register link seprately
def log_in():
    if request.method == 'POST':
        users.append(request.form['user-name'])
        session['username'] = users[-1]

        if request.form['mode'] == "Standard":
            return redirect("register")
        else:
            return redirect("recon")
    else:
        if 'username' in session and session['username'] in users:

            return render_template("log_in.html")
        else:

            return render_template("log_in.html")


@app.route('/register')  # This needs to point to the login screen and then we can use the register link seprately
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


@app.route('/acquire')
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


@app.route('/analyze')
def on_analyze():
    return render_template('analyze.html')


@app.route('/recon')
def on_recon():
    return render_template('recon.html')


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
    # read payload and convert it to dictionary
    payload = request.data
    payload = json.loads(payload.decode('utf8'))

    formName = payload.get('formName')
    # Do registration and save to database
    if formName == 'reg':
        del payload['formName']

        pat_id = payload.get('patid')
        session['patid'] = pat_id
        query_dict = {
            "patid": pat_id,
        }
        rows = reg.reuse(query_dict)
        # print((rows))

        if (rows):
            print('Subject is already registered with PATID: ' + pat_id)
        else:
            status = reg.consume(payload)

    if formName == 'acq':
        pat_id = session['patid']
        query_dict = {
            "patid": pat_id,
        }

        rows = reg.reuse(query_dict)
        bsim.run_blochsim(seqinfo=payload, phtinfo=rows[0][0])  # phtinfo just needs to be 1 string

    result = ''
    return result


if __name__ == '__main__':
    # run!
    # app.run(host='0.0.0.0', debug=True)
    app.run(host='0.0.0.0', debug=True)
