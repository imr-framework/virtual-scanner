from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent

COMS_PATH = ROOT_PATH / 'coms'
SERVER_ANALYZE_PATH = COMS_PATH / 'server' / 'ana'
SERVER_PATH = ROOT_PATH / 'server'

# GT
SERVER_SIM_BLOCH_PATH = SERVER_PATH / 'simulation'/ 'bloch'
SERVER_SIM_OUTPUTS_PATH = SERVER_PATH / 'simulation' / 'outputs'
COMS_SIM_OUTPUTS_PATH = COMS_PATH / 'coms_ui' / 'static' / 'acq' / 'outputs'
#


STATIC_ANALYZE_PATH = Path('static') / 'ana'
USER_UPLOAD_FOLDER = ROOT_PATH / 'coms' / 'coms_ui' / 'static' / 'user_uploads'



