#!flask/bin/python

import sys
from flask import Flask, render_template, request, redirect, Response
import random, json

app = Flask(__name__)


@app.route('/')  #This needs to point to the login screen and then we can use the register link seprately
def output():
    # serve register template
    return render_template('register.html')

@app.route('/acquire.html')
def acq():
    # serve index template
    return render_template('acquire.html')


# @app.route('/acquire.html')
# def analyze():
#     # serve index template
#     return render_template('acquire.html')

@app.route('/receiver', methods=['POST'])
def worker():
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
