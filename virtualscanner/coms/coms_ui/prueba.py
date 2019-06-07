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
   tx to client
   rx from client

Unit Test app
-------------
     utest_coms_flask
Author: Sairam Geethanath , Modified by: Marina Manso Jimeno
Date: 03/22/2019
Version 0.0
Copyright of the Board of Trustees of  Columbia University in the City of New York
"""
from flask import Flask, render_template, request, session

# Define the location of template and static folders
# template_dir = os.path.abspath('../templates')
# static_dir=os.path.abspath('../static')

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

users = []
app.secret_key = 'Clave de sesión'


@app.route('/', methods=['POST',
                         'GET'])  # This needs to point to the login screen and then we can use the register link seprately
def log_in():
    if request.method == 'POST':
        users.append(request.form['user-name'])
        session['username'] = users[-1]

        return render_template("home_page.html")
    else:
        if 'username' in session and session['username'] in users:

            return render_template("home_page.html")
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
                rx payload from the client
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
