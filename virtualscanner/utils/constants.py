from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent

COMS_PATH = ROOT_PATH / 'coms'
RECON_PATH = ROOT_PATH / 'recon'
SERVER_ANALYZE_PATH = COMS_PATH / 'server' / 'ana'
SERVER_PATH = ROOT_PATH / 'server'
STATIC_ANALYZE_PATH = Path('static') / 'ana'
USER_UPLOAD_FOLDER = ROOT_PATH / 'coms' / 'coms_ui' / 'static' / 'user_uploads'

