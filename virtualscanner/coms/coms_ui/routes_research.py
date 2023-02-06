
from flask import Flask, flash, render_template, session, redirect, url_for, request
from __main__ import app, db, socketio  #, login_manager
import numpy as np
# B0
from virtualscanner.coms.coms_ui.forms import HalbachForm
from virtualscanner.server.b0.b0_worker import b0_halbach_worker, b0_plot_worker

@app.route('/research',methods=['POST','GET'])
def research():
    return render_template('research.html')

@app.route('/research/sequence',methods=['POST','GET'])
def sequence():
    return render_template('sequence.html')

@app.route('/research/b0',methods=['POST','GET'])
def b0():
    b0form = HalbachForm()
    return render_template('b0.html',template_form = b0form)

@socketio.on('B0 run Halbach')
def run_halbach_sim(payload):
    print(payload)
    # Parse
    radii_array_str = payload['innerRadii'].split(',')
    innerRingRadii = np.array([float(radius) for radius in radii_array_str])*1e-3
    num_array_str = payload['innerNum'].split(',')
    innerNumMagnets = np.array([int(num) for num in num_array_str])
    numRings = int(payload['numRings'])
    ringSep = float(payload['ringSep'])*1e-3
    DSV = float(payload['dsv'])*1e-3

    # Get simulated data
    minTracker, coordinateAxis, maskedField, output_text = b0_halbach_worker(innerRingRadii, innerNumMagnets, numRings, ringSep, DSV)
    # Update session


    # Print program output
    socketio.emit('B0 print Halbach program output',{'output':output_text})

    # Plot magnetic field
    j1 = b0_plot_worker(maskedField, int(np.floor(np.size(maskedField,0)/2)),
                                     int(np.floor(np.size(maskedField,1)/2)),
                                     int(np.floor(np.size(maskedField,2)/2)))
    print('B0 plot generated!')
    socketio.emit('Update B0 plot',{'graphData':j1})


# TODO
@socketio.on('Update Halbach plot')
def update_halbach_plot(xslice, yslice, zslice):
    # Update view using session data
    #maskedField = session['b0']['b0_simulated']
    socketio.emit('Update Halbach plot',{})