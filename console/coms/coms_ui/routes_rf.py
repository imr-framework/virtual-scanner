
from flask import Flask, flash, render_template, session, redirect, url_for, request
from __main__ import app, db, socketio  #, login_manager
import numpy as np
import os
import json
import math
# B0
from console.utils.constants import SEQ_DATA_PATH

from console.coms.coms_ui.forms import HalbachForm, SequenceForm
from console.server.b0.b0_worker import b0_halbach_worker, b0_plot_worker, b0_3dplot_worker, b0_rings_worker,\
                                               b0_eval_field_any
from console.utils import constants
from console.utils.helpers import update_session_subdict
from scipy.io import savemat, loadmat
from console.coms.coms_ui.forms import RFForm

from console.server.simulation.rf_sim.rf_worker import *

@app.route('/research/rf',methods=['POST','GET'])
def rf_sim_page():
    # Sequence uploading
    rf_form = RFForm()
    return render_template('rf.html',template_form=rf_form)

@socketio.on('Update session variable rf')
def update_session_parameters(info):
    print(info)
    update_session_subdict(session,'rf',{info['id']:info['value']})
    print(session['rf'])
    return

@socketio.on('Display RF')
def display_rf_pulse(payload):
    j1 = rf_display_worker(payload)
    socketio.emit('Deliver RF pulse', {'graph':j1})
    return

@socketio.on('Simulate RF')
def simulate_rf_pulse(payload):
    j2, j3 = rf_simulate_worker(payload)
    socketio.emit('Deliver RF profile',{'graph-profile':j2,'graph-evol':j3})
    return

@socketio.on('Generate code from RF settings')
def get_rf_code(payload):
    code = generate_rf_code(payload)
    socketio.emit('Deliver RF code', {'code': code})
    return

