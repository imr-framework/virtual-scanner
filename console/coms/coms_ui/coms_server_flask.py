# Copyright of the Board of Trustees of Columbia University in the City of New York

import os, signal
from flask_login import LoginManager, UserMixin, login_required, login_user, current_user, logout_user
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy

if __name__ == '__main__':
    import sys
    script_path = os.path.abspath(__file__)
    SEARCH_PATH = script_path[:script_path.index('console')]
    sys.path.insert(0, SEARCH_PATH)

from pathlib import Path
from flask import Flask, render_template, request, redirect, session
from flask_session import Session
from console.utils import constants

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
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/test.db'  # ROOT_PATH / "subject.db" #
Session(app)
db = SQLAlchemy(app)
#
# login_manager = LoginManager()
# login_manager.init_app(app)

socketio = SocketIO(app, manage_session=False)

users = []
n_acqs = 0

app.secret_key = 'Session_key'


def launch_console():
    """
    Runs the server in the specified machine's local network address.
    """
    import console.coms.coms_ui.routes_b0
    import console.coms.coms_ui.routes_original
    import console.coms.coms_ui.routes_twin
    import console.coms.coms_ui.routes_research
    import console.coms.coms_ui.routes_sequence
    import console.coms.coms_ui.routes_rf
    # app.run(host='0.0.0.0', debug=True)
    socketio.run(app, debug=True, host="0.0.0.0", allow_unsafe_werkzeug=True)


def kill_console():
    pid = os.getpid()
    os.kill(pid, signal.SIGTERM)


if __name__ == '__main__':
    launch_console()
