from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent

COMS_PATH = ROOT_PATH / 'coms'
RECON_PATH = ROOT_PATH / 'recon'
SERVER_ANALYZE_PATH = ROOT_PATH / 'server' / 'ana'  # modified by EQ to fix ana bug
COMS_ANALYZE_PATH =  COMS_PATH / 'coms_ui'  # modified by EQ to fix ana bug
SERVER_PATH = ROOT_PATH / 'server'
IMG_SAR_PATH = COMS_ANALYZE_PATH/'static'/'RF'/'tx'/'SAR/'
# GT
SERVER_SIM_BLOCH_PATH = SERVER_PATH / 'simulation'/ 'bloch'
SERVER_SIM_OUTPUTS_PATH = SERVER_PATH / 'simulation' / 'outputs'
SERVER_RX_PATH = SERVER_PATH / 'rx'
COMS_RX_OUTPUTS_PATH = COMS_PATH / 'coms_ui' / 'static' / 'Rx' / 'outputs'
COMS_RX_INPUTS_PATH = COMS_PATH / 'coms_ui' / 'static' / 'Rx' / 'inputs'
COMS_SIM_OUTPUTS_PATH = COMS_PATH / 'coms_ui' / 'static' / 'acq' / 'outputs'

#


STATIC_ANALYZE_PATH = Path('static') / 'ana'
STATIC_ACQUIRE_PATH = Path('static') / 'acq'
STATIC_RF_PATH = Path('static') / 'rf'
STATIC_RFTX_PATH = STATIC_RF_PATH/'tx'
SAR_PATH = SERVER_PATH/'rf'/ 'tx'/'SAR_calc'
STATIC_RX_PATH = Path('static') / 'Rx' / 'outputs'
STATIC_RECON_PATH = Path('static') / 'Recon'
USER_UPLOAD_FOLDER = ROOT_PATH / 'coms' / 'coms_ui' / 'static' / 'user_uploads'

RECON_PATH = ROOT_PATH / 'recon'

