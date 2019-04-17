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
#import os
import sys
from flask import Flask, render_template, request, redirect, Response
import random, json
import register as reg

# Define the location of template and static folders
#template_dir = os.path.abspath('../templates')
#static_dir=os.path.abspath('../static')

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.route('/', methods =['POST','GET'])  # This needs to point to the login screen and then we can use the register link seprately
def log_in():


    if request.method == 'POST':
        users.append(request.form['user-name'])
        session['username']= users[-1]

        if request.form['mode'] == "Standard":
            return redirect("register")
        else:
            return redirect("recon")
    else:
        if 'username' in session  and session['username'] in users:

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
        status = reg.consume(payload)
        payload = reg.reuse(payload)
        reg_payload = payload #For GT to use for her implementation, TODO: Refine this for integrated implementation

    result = ''
    return result



if __name__ == '__main__':
    # run!
    # app.run(host='0.0.0.0', debug=True)
    app.run(host='0.0.0.0', debug=True)
