# Copyright of the Board of Trustees of Columbia University in the City of New York

import os, signal
from flask_login import LoginManager, UserMixin, login_required, login_user, current_user, logout_user
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy


if __name__ == '__main__':
    import sys

    script_path = os.path.abspath(__file__)
    SEARCH_PATH = script_path[:script_path.index('virtualscanner')]
    sys.path.insert(0, SEARCH_PATH)

from pathlib import Path

from flask import Flask, render_template, request, redirect, session
from flask_session import Session


from virtualscanner.utils import constants

CURRENT_PATH = Path(__file__).parent
ROOT_PATH = constants.ROOT_PATH
UPLOAD_FOLDER = constants.COMS_UI_STATIC_USER_UPLOAD_PATH
SERVER_ANALYZE_PATH = constants.SERVER_ANALYZE_PATH
STATIC_ANALYZE_PATH = constants.COMS_UI_STATIC_ANALYZE_PATH


app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TESTING'] = True
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)
db = SQLAlchemy(app)
#
# login_manager = LoginManager()
# login_manager.init_app(app)

socketio = SocketIO(app, manage_session=False)


users = []
n_acqs = 0

app.secret_key = 'Session_key'


def launch_virtualscanner():
    """
    Runs the server in the specified machine's local network address.
    """
    import virtualscanner.coms.coms_ui.routes_research
    import virtualscanner.coms.coms_ui.routes_original
    import virtualscanner.coms.coms_ui.routes_twin
    #app.run(host='0.0.0.0', debug=True)
    socketio.run(app, debug=True, host="0.0.0.0")

def kill_virtualscanner():
    pid = os.getpid()
    os.kill(pid, signal.SIGTERM)

if __name__ == '__main__':
    launch_virtualscanner()
