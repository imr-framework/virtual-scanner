# Copyright of the Board of Trustees of Columbia University in the City of New York


if __name__ == '__main__':
    import os
    import sys

    script_path = os.path.abspath(__file__)
    SEARCH_PATH = script_path[:script_path.index('virtual-scanner') + len('virtual-scanner') + 1]
    sys.path.insert(0, SEARCH_PATH)

from pathlib import Path

from flask import Flask, render_template, request, redirect, session
from werkzeug.utils import secure_filename

from virtualscanner.server.ana import T2_mapping as T2_mapping, T1_mapping as T1_mapping, ROI_analysis as ROI_analysis
from virtualscanner.server.recon.drunck.reconstruct import main
from virtualscanner.server.registration import register as reg
from virtualscanner.server.rf.rx import caller_script_Rx as Rxfunc
from virtualscanner.server.rf.tx.SAR_calc import SAR_calc_main as SAR_calc_main
from virtualscanner.server.simulation.bloch import caller_script_blochsim as bsim
from virtualscanner.utils import constants

CURRENT_PATH = Path(__file__).parent
ROOT_PATH = constants.ROOT_PATH
UPLOAD_FOLDER = constants.COMS_UI_STATIC_USER_UPLOAD_PATH
SERVER_ANALYZE_PATH = constants.SERVER_ANALYZE_PATH
STATIC_ANALYZE_PATH = constants.COMS_UI_STATIC_ANALYZE_PATH

ALLOWED_EXTENSIONS = {'seq', 'jpg'}

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

users = []
n_acqs = 0

app.secret_key = 'Session_key'


@app.route('/', methods=['POST',
                         'GET'])  # This needs to point to the login screen and then we can use the register link seprately
def log_in():
    """
    Renders the log-in html page on the web and requests user log-in information (e-mail) and choice of user mode
    (Standard/Advanced).

    Returns
    -------
    AppContext
        | Redirects to register page if user-name exists and Standard mode is selected
        |    OR
        | Redirects to recon page if user-name exists and Advanced mode is selected
        |    OR
        | Renders log-in template
    """
    session.clear()
    session['acq_out_axial'] = []
    session['acq_out_sagittal'] = []
    session['acq_out_coronal'] = []
    if request.method == 'POST':
        # users.append(request.form['user-name'])
        # session['username'] = users[-1]
        session['username'] = request.form['user-name']
        if session['username'] == "":
            return render_template("log_in.html")
        if request.form['mode'] == "Standard":
            return redirect("register")
        else:
            return redirect("recon")
    else:
        return render_template("log_in.html")


# This needs to point to the login screen and then we can use the register link separately
@app.route('/register', methods=['POST', 'GET'])
def on_register():
    """
    Renders the registration html page on the web.

    Returns
    -------
    AppContext
        | Renders register page
        |    OR
        | Redirects to register success page if registration occurs

    """
    if 'acq' in session and 'reg_success' not in session:
        session.pop('acq')
        message = 1
        return render_template('register.html', msg=message)

    if 'ana_load' in session and 'reg_success' not in session:
        session.pop('ana_load')
        message = 1
        return render_template('register.html', msg=message)

    if 'reg_success' in session:
        return redirect('register_success')
    else:
        return render_template('register.html')


@app.route('/register_success', methods=['POST', 'GET'])
def on_register_success():
    """
    Renders the registration html page on the web with a success message when registration occurs.

    Returns
    -------
    AppContext
        | Renders register page with registration success message
        | May also pass variables:
        | success : int
        |    Either 1 or 0 depending on registration success
        | payload : dict
        |    Form inputted values sent back to display together with template

    """
    return render_template('register.html', success=session['reg_success'], payload=session['reg_payload'])


@app.route('/acquire', methods=['POST', 'GET'])
def on_acq():
    """
    Renders the acquire html page on the web.

    Returns
    -------
    AppContext
        | Renders acquire template
        | May also pass variables:
        | success : int
        |    Either 1 or 0 depending on acquisition success
        | axial : list
        |    File names for generated axial images
        | sagittal : list
        |    File names for generated sagittal images
        | coronal : list
        |    File names for generated coronal images
        | payload : dict
        |    Form inputted values sent back to display together with template
    """
    if 'acq' in session:

        return render_template('acquire.html', success=session['acq'], axial=session['acq_out_axial'],
                               sagittal=session['acq_out_sagittal'], coronal=session['acq_out_coronal'],
                               payload=session['acq_payload'])
    else:
        return render_template('acquire.html')


@app.route('/analyze', methods=['POST', 'GET'])
def on_analyze():
    """
    Renders the analyze html page on the web.

    Returns
    -------
    AppContext
        | Renders analyze template
        | May also pass variables:
        | roi_success/map_success/load_success : int
        |   Either 1 or 0 depending on analyze steps success
        | payload1/payload2/payload3 : dict
        |   Form inputted values and output results sent back to display together with template
    """
    if 'ana_load' in session:
        if 'ana_map' in session:
            if 'ana_roi' in session:
                return render_template('analyze.html', roi_success=session['ana_roi'], payload3=session['ana_payload3'],
                                       map_success=session['ana_map'], load_success=session['ana_load'],
                                       payload1=session['ana_payload1'], payload2=session['ana_payload2'])
            else:
                return render_template('analyze.html', map_success=session['ana_map'], load_success=session['ana_load'],
                                       payload1=session['ana_payload1'], payload2=session['ana_payload2'])
        else:
            return render_template('analyze.html', load_success=session['ana_load'], payload1=session['ana_payload1'])

    else:
        return render_template('analyze.html')


@app.route('/tx', methods=['POST', 'GET'])
def on_tx():
    """
    Renders the tx html page on the web.

    Returns
    -------
    AppContext
        | Renders tx template
        | May also pass variables:
        | success : int
        |   Either 1 or 0 depending on tx success
        | payload : dict
        |   Form inputted values and output results sent back to display together with template
    """
    if 'tx' in session:
        return render_template('tx.html', success=session['tx'], payload=session['tx_payload'])
    else:
        return render_template('tx.html')


@app.route('/rx', methods=['POST', 'GET'])
def on_rx():
    """
    Renders the rx html page on the web.

    Returns
    -------
    AppContext
        | Renders rx template
        | May also pass variables:
        | success : int
        |   Either 1 or 0 depending on rx success
        | payload : dict
        |   Form inputted values and output results sent back to display together with template
    """
    if 'rx' in session:
        return render_template('rx.html', success=session['rx'], payload=session['rx_payload'])
    else:
        return render_template('rx.html')


@app.route('/recon', methods=['POST', 'GET'])
def on_recon():
    """
    Renders the recon html page on the web.

    Returns
    -------
    AppContext
        | Renders recon template
        | May also pass variables:
        | success : int
        |   Either 1 or 0 depending on recon success
        | payload : dict
        |   Form inputted values and output results sent back to display together with template
    """
    if 'recon' in session:
        return render_template('recon.html', success=session['recon'], payload=session['recon_payload'])
    else:
        return render_template('recon.html')


def allowed_file(filename):
    """
    Checks that the file extension is within the application allowed extensions.

    Parameters
    ----------
    filename : str
        Uploaded file name

    Returns
    -------
    bool
        Allowed or not allowed extension
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/receiver', methods=['POST', 'GET'])
def worker():
    """
    Receives form inputs from the templates and applies the server methods.

    Returns
    -------
    AppContext
        Either renders templates or redirects to other templates
    """
    # read payload and convert it to dictionary

    if request.method == 'POST':
        payload = request.form.to_dict()

        # Registration
        if request.form['formName'] == 'reg':

            if payload['subjecttype'] == "Subject":
                return redirect('register')

            print(payload)
            session['reg_success'] = 1
            session['reg_payload'] = payload

            del payload['formName']

            # Currently only metric system since only phantom registration is possible. Fix this for future releases.
            del payload['height-unit']
            del payload['weight-unit']
            del payload['inches']

            pat_id = payload.get('patid')
            session['patid'] = pat_id
            query_dict = {
                "patid": pat_id,
            }
            rows = reg.reuse(query_dict)
            # print((rows))d

            if (rows):
                print('Subject is already registered with PATID: ' + pat_id)
            else:
                status = reg.consume(payload)

            return redirect('register_success')

        elif request.form['formName'] == 'new-reg':
            session.pop('reg_success')

            return redirect('register')
        # ACQUIRE
        elif request.form['formName'] == 'acq':

            session['acq'] = 0

            if "patid" not in session:  # Please register first
                return redirect('register')

            pat_id = session['patid']
            query_dict = {
                "patid": pat_id,
            }

            rows = reg.reuse(query_dict)  #
            print(rows)

            # session['acq'] = 0
            session['acq_payload'] = payload
            print(payload)

            progress = bsim.run_blochsim(seqinfo=payload, phtinfo=rows[0][0],
                                         pat_id=pat_id)  # phtinfo just needs to be 1 string
            sim_result_path = constants.COMS_PATH / 'coms_ui' / 'static' / 'acq' / 'outputs' / session['patid']

            while (os.path.isdir(sim_result_path) is False):
                pass

            if progress == 1:
                session['acq'] = 1

                STATIC_ACQUIRE_PATH_REL = constants.COMS_UI_STATIC_ACQUIRE_PATH.relative_to(CURRENT_PATH)
                im_path_from_template = STATIC_ACQUIRE_PATH_REL / 'outputs' / session['patid']

                imgpaths = os.listdir(sim_result_path)
                complete_path = [str(im_path_from_template / iname) for iname in imgpaths]
                Z_acq = []
                X_acq = []
                Y_acq = []

                for indx in range(len(complete_path)):
                    if payload['selectedSeq'] == 'GRE':
                        pos = complete_path[indx].find('_', 30, ) + 1  #
                    else:
                        pos = complete_path[indx].find('_', 29, ) + 1  #

                    sl_orientation = complete_path[indx][pos]
                    if sl_orientation == 'Z':
                        Z_acq.append(complete_path[indx])
                    elif sl_orientation == 'X':
                        X_acq.append(complete_path[indx])
                    elif sl_orientation == 'Y':
                        Y_acq.append(complete_path[indx])

                session['acq_out_axial'] = Z_acq
                session['acq_out_sagittal'] = X_acq
                session['acq_out_coronal'] = Y_acq

                return redirect('acquire')

        # Analyze
        elif request.form['formName'] == 'ana':

            if 'original-data-opt' in payload:

                if 'ana_load' in session:
                    session.pop('ana_load')
                    if 'ana_map' in session:
                        session.pop('ana_map')
                        if 'ana_roi' in session:
                            session.pop('ana_roi')

                session['ana_load'] = 1

                if "patid" not in session:  # Please register first
                    return redirect('register')

                if payload['original-data-opt'] == 'T1':
                    folder_path = STATIC_ANALYZE_PATH / 'inputs' / 'T1_original_data'
                elif payload['original-data-opt'] == 'T2':
                    folder_path = STATIC_ANALYZE_PATH / 'inputs' / 'T2_original_data'

                filenames_in_path = os.listdir(folder_path)
                STATIC_ANALYZE_PATH_REL = constants.COMS_UI_STATIC_ANALYZE_PATH.relative_to(CURRENT_PATH)
                original_data_path = [
                    str(STATIC_ANALYZE_PATH_REL / 'inputs' / (payload['original-data-opt'] + '_original_data') / iname)
                    for
                    iname in filenames_in_path]
                payload['data-path'] = original_data_path

                session['ana_payload1'] = payload


            elif 'map-form' in payload:
                STATIC_ANALYZE_PATH_REL = constants.COMS_UI_STATIC_ANALYZE_PATH.relative_to(CURRENT_PATH)
                session['ana_map'] = 1

                if payload['TI'] == "":
                    server_od_path = SERVER_ANALYZE_PATH / 'inputs' / 'T2_orig_data'
                    map_name, dicom_path = T2_mapping.main(server_od_path, payload['TR'], payload['TE'],
                                                           session['patid'])
                else:
                    server_od_path = SERVER_ANALYZE_PATH / 'inputs' / 'T1_orig_data'
                    map_name, dicom_path = T1_mapping.main(server_od_path, payload['TR'], payload['TE'], payload['TI'],
                                                           session['patid'])

                payload['dicom_path'] = str(dicom_path)
                payload['map_path'] = str(STATIC_ANALYZE_PATH_REL / 'outputs' / session['patid'] / map_name)
                session['ana_payload2'] = payload


            elif 'roi-form' in payload:

                # payload['map-type'] = 'T1'
                STATIC_ANALYZE_PATH_REL = constants.COMS_UI_STATIC_ANALYZE_PATH.relative_to(CURRENT_PATH)
                session['ana_roi'] = 1
                dicom_map_path = session['ana_payload2']['dicom_path']
                if 'T1' in dicom_map_path:
                    payload['map-type'] = 'T1'
                elif 'T2' in dicom_map_path:
                    payload['map-type'] = 'T2'

                roi_result_filename = ROI_analysis.main(dicom_map_path, payload['map-type'], payload['map-size'],
                                                        payload['map-FOV'], session['patid'])

                roi_result_path = STATIC_ANALYZE_PATH_REL / 'outputs' / session['patid'] / roi_result_filename

                payload['roi_path'] = str(roi_result_path)
                session['ana_payload3'] = payload

            return redirect('analyze')

        # Advance Mode
        # tx
        elif request.form['formName'] == 'tx':
            print(payload)
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)

                upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(upload_path)

                filename = filename[:-4] + '.seq'

                if (filename != 'rad2d.seq'):
                    if (os.path.isfile(constants.SERVER_PATH / 'rf' / 'tx' / 'SAR_calc' / 'assets' / filename)):
                        os.remove(constants.SERVER_PATH / 'rf' / 'tx' / 'SAR_calc' / 'assets' / filename)
                    dest = str(constants.SERVER_PATH / 'rf' / 'tx' / 'SAR_calc' / 'assets' / filename)
                    os.rename(upload_path, dest)

                # os.rename(upload_path, constants.SERVER_PATH / 'rf' / 'tx' / 'SAR_calc' / 'assets' / filename)

                output = SAR_calc_main.payload_process(filename)

                session['tx'] = 1
                STATIC_RFTX_PATH_REL = constants.COMS_UI_STATIC_RFTX_PATH.relative_to(CURRENT_PATH)
                output['plot_path'] = str(STATIC_RFTX_PATH_REL / 'SAR' / output['filename'])
                session['tx_payload'] = output
                return redirect('tx')
        # rx
        elif request.form['formName'] == 'rx':

            signals_filename, recon_filename, orig_im_path = Rxfunc.run_Rx_sim(payload)

            COMS_UI_STATIC_RFRX_PATH_REL = constants.COMS_UI_STATIC_RX_OUTPUT_PATH.relative_to(CURRENT_PATH)

            payload['signals_path'] = str(COMS_UI_STATIC_RFRX_PATH_REL / signals_filename)
            payload['recon_path'] = str(COMS_UI_STATIC_RFRX_PATH_REL / recon_filename)

            session['rx'] = 1
            session['rx_payload'] = payload
            return redirect('rx')

        # recon
        elif request.form['formName'] == 'recon':
            file = request.files['file']
            filename = secure_filename(file.filename)

            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_path)

            STATIC_RECON_PATH_REL = constants.COMS_UI_STATIC_RECON_PATH.relative_to(CURRENT_PATH)
            out_path_form_template = STATIC_RECON_PATH_REL / 'outputs'

            if payload['DL-type'] == "GT":
                out1, out2, out3 = main(input_path, payload['DL-type'])
                payload['output'] = [str(out_path_form_template / out1), str(out_path_form_template / out2),
                                     str(out_path_form_template / out3)]
            else:
                out1, out2 = main(input_path, payload['DL-type'])
                payload['output'] = [out_path_form_template / out1, out_path_form_template / out2]

            session['recon'] = 1
            session['recon_payload'] = payload

            return redirect('recon')


def launch_virtualscanner():
    """
    Runs the server in the specified machine's local network address.
    """
    app.run(host='0.0.0.0', debug=True)


if __name__ == '__main__':
    launch_virtualscanner()
