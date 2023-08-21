
from flask import Flask, flash, render_template, session, redirect, url_for, request
from __main__ import app, db, socketio  #, login_manager
import numpy as np
# B0
from console.coms.coms_ui.forms import HalbachForm
from console.server.b0.b0_worker import b0_halbach_worker, b0_plot_worker, b0_3dplot_worker, b0_rings_worker,\
                                               b0_eval_field_any
from console.utils import constants
from console.utils.helpers import update_session_subdict
from scipy.io import savemat, loadmat

@app.route('/research',methods=['POST','GET'])
def research():
    return render_template('research.html')
