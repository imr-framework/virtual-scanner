from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent

# coms_ui
COMS_PATH = ROOT_PATH / 'coms'
COMS_UI_PATH = COMS_PATH / 'coms_ui'
COMS_UI_STATIC_USER_UPLOAD_PATH = ROOT_PATH / 'coms' / 'coms_ui' / 'static' / 'user_uploads'
COMS_UI_STATIC_ANALYZE_PATH = Path('static') / 'ana'
COMS_UI_STATIC_ACQUIRE_PATH = Path('static') / 'acq'
COMS_UI_STATIC_RF_PATH = Path('static') / 'rf'
COMS_UI_STATIC_RFTX_PATH = COMS_UI_STATIC_RF_PATH / 'tx'
COMS_UI_STATIC_RX_PATH = Path('static') / 'Rx' / 'outputs'
COMS_UI_STATIC_RECON_PATH = Path('static') / 'recon'
COMS_UI_STATIC_RX_OUTPUT_PATH = COMS_PATH / 'coms_ui' / 'static' / 'Rx' / 'outputs'
COMS_UI_STATIC_RX_INPUT_PATH = COMS_PATH / 'coms_ui' / 'static' / 'Rx' / 'inputs'
COMS_UI_SIM_OUTPUT_PATH = COMS_PATH / 'coms_ui' / 'static' / 'acq' / 'outputs'

# server
SERVER_PATH = ROOT_PATH / 'server'

# server.ana

# server.quant_analysis
SERVER_ANALYZE_PATH = SERVER_PATH / 'ana'

# server.recon
RECON_PATH = ROOT_PATH / 'server' / 'recon'
RECON_ASSETS_PATH = RECON_PATH / 'drunck' / 'assets'

# server.registration

# server.rf
RF_SAR_STATIC_IMG_PATH = COMS_UI_PATH / 'static' / 'RF' / 'tx' / 'SAR/'
RF_SAR_PATH = SERVER_PATH / 'rf' / 'tx' / 'SAR_calc'

# server.rx
RX_PATH = SERVER_PATH / 'rx'

# server.simulation
SERVER_SIM_BLOCH_PATH = SERVER_PATH / 'simulation' / 'bloch'
SERVER_SIM_OUTPUT_PATH = SERVER_PATH / 'simulation' / 'outputs'
