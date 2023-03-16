
from flask import Flask, flash, render_template, session, redirect, url_for, request
from __main__ import app, db, socketio  #, login_manager
import numpy as np
# B0
from virtualscanner.coms.coms_ui.forms import HalbachForm
from virtualscanner.server.b0.b0_worker import b0_halbach_worker, b0_plot_worker, b0_3dplot_worker, b0_rings_worker,\
                                               b0_eval_field_any
from virtualscanner.utils import constants
from virtualscanner.utils.helpers import update_session_subdict
from scipy.io import savemat, loadmat


@app.route('/research/b0',methods=['POST','GET'])
def b0():
    b0form = HalbachForm()
    return render_template('b0.html',template_form = b0form)

@app.route('/research/simulate',methods=['POST','GET'])
def simulate():
    return render_template('simulate.html',template_form = [])

@socketio.on('B0 run Halbach')
def run_halbach_sim(payload):
    # Parse
    radii_array_str = payload['innerRadii'].split(',')
    innerRingRadii = np.array([float(radius) for radius in radii_array_str])*1e-3
    num_array_str = payload['innerNum'].split(',')
    innerNumMagnets = np.array([int(num) for num in num_array_str])
    numRings = int(payload['numRings'])
    ringSep = float(payload['ringSep'])*1e-3
    DSV = float(payload['dsv'])*1e-3
    max_num_gen = int(payload['maxGen'])
    resolution = float(payload['resolution'])

    # Get simulated data
    minTracker, coordinateAxis, maskedField,\
    output_text, best_vector, \
    ring_position_symmetry, simDimensions = b0_halbach_worker(innerRingRadii, innerNumMagnets,
                                                                                          numRings, ringSep, DSV,
                                                                                          max_num_gen, resolution)
    # TODO use sim defaults here!

    outerRingRadii = innerRingRadii + 21 * 1e-3  # Fixed radius difference between inner and outer at 21 mm
    outerNumMagnets = innerNumMagnets + 7 # Fixed number of magnets difference between inner and outer at 7



    # Update session
    update_session_subdict(session, 'b0', {'best_vector': best_vector,
                                           'masked_field':maskedField,
                                           'coordinates': [coordinateAxis],
                                           'ring_position_symmetry': ring_position_symmetry,
                                           'inner_num_magnets': innerNumMagnets,
                                           'inner_ring_radii': innerRingRadii,
                                           'outer_num_magnets': outerNumMagnets,
                                           'outer_ring_radii': outerRingRadii,
                                           'resolution': resolution,
                                           'sim_dimensions': simDimensions,
                                           'dsv': DSV})

    # Save data
    # savemat(constants.B0_DATA_PATH / 'halbach.mat',{'best_vector': best_vector,
    #                                                 'ring_position_symmetry':ring_position_symmetry,
    #                                                 'innerNumMagnets':innerNumMagnets,
    #                                                 'innerRingRadii':innerRingRadii,
    #                                                 'resolution':resolution,
    #                                                 'simDimensions': simDimensions})

    # Print program output
    socketio.emit('B0 print Halbach program output',{'best_vector':best_vector.__str__()})

    # Plot magnetic field
    mid_indices = [int(np.floor(np.size(maskedField,q)/2)) for q in range(3)]
    j1 = b0_plot_worker(maskedField, [coordinateAxis], mid_indices[0], mid_indices[1], mid_indices[2])

    print('B0 plot generated!')
    socketio.emit('Update B0 plot',{'graphData':j1, 'x':mid_indices[0],'y':mid_indices[1],'z':mid_indices[2]})


@socketio.on('Update Halbach plot')
def update_halbach_plot(xslice, yslice, zslice):
    # Update view using session data
    #maskedField = session['b0']['b0_simulated']
    socketio.emit('Update Halbach plot',{})

@socketio.on('Update B0 session')
def update_b0_session(params):
    print(params)
    update_session_subdict(session,'b0',params)

@socketio.on('Get 3D plot')
def get_3d_plot():
    axis = session['b0']['opt-3d']
    coordinates = session['b0']['coordinates']

    # Get maskedField ...
    # TODO recreate field from best vector... or retrieve from data mat? or retrieve from session?
    j2 = b0_3dplot_worker(session['b0']['masked_field'], coordinates, axis)
    socketio.emit('Update 3D plot',{'graphData':j2})


@socketio.on('Get rings plot')
def get_rings_plot():
    # TODO must make this worker account for different outer ring radii!!
    j3 = b0_rings_worker(session['b0']['inner_ring_radii'],
                         session['b0']['outer_ring_radii'],
                         session['b0']['ring_position_symmetry'],
                         session['b0']['dsv'],
                         session['b0']['best_vector'])

    socketio.emit('Update rings plot', {'graphData':j3})

@socketio.on("Update Halbach slices")
def update_halbach_slices(info):
    # Update masked_field with new dsv
    d = info['dsv_display']
    masked_field, coordinates = b0_eval_field_any(diameter=d, info=session['b0'],temperature=info['temperature'])
    print('Size of masked field: ', masked_field.shape)
    j1 = b0_plot_worker(masked_field, [coordinates], int(info['x']),int(info['y']),int(info['z']))
    update_session_subdict(session,'b0',{'masked_field': masked_field, 'coordinates': [coordinates]})

    socketio.emit('Update B0 plot',
                  {'graphData': j1, 'x':int(info['x']),'y':int(info['y']),'z':int(info['z'])})

@socketio.on("Save B0 session to data")
def save_b0_session():
    # Save data
    savemat(constants.B0_DATA_PATH / 'halbach.mat', session['b0'])
    socketio.emit('B0 session saved')

@socketio.on("Load B0 session from data")
def load_b0_session(info):
    print("B0 load type: ", info['type'])
    if info['type'] == 'prevsim':
        try:
            data = loadmat(constants.B0_DATA_PATH / 'halbach.mat')
        except:
            print("Failed to load halbach.mat")
            socketio.emit('B0 session loading failed')
    elif info['type'] == 'default':
        try:
            data = loadmat(constants.B0_DATA_PATH / 'default.mat')

        except:
            print("Failed to load default.mat")
            socketio.emit('B0 session loading failed')

    update_session_subdict(session,'b0',data)

    print('Session just before updating slices!')
    print(session['b0'])
    update_halbach_slices(session['b0'])

    socketio.emit('B0 session loaded')
