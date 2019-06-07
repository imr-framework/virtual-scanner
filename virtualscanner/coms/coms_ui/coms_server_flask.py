"""
This script runs the server based on the Flask package and hosts the GUIs on the web
Parameters
----------
    payload,default = None

Returns
-------
    payload or status on ping from clients

Performs
--------
   tx to client
   rx from client

Unit Test app
-------------
     utest_coms_flask
Author: Sairam Geethanath , Modified by: Marina Manso Jimeno
Date: 03/22/2019
Version 0.0
Copyright of the Board of Trustees of  Columbia University in the City of New York
"""

import os
import time

from flask import Flask, render_template, request, redirect, session
from werkzeug.utils import secure_filename

import virtualscanner.server.registration.register as reg
from virtualscanner.server.ana import T2_mapping as T2_mapping, T1_mapping as T1_mapping, ROI_analysis as ROI_analysis
from virtualscanner.server.recon.drunck.reconstruct import main
from virtualscanner.server.rf.tx.SAR_calc import SAR_calc_main as SAR_calc_main
from virtualscanner.server.rx import caller_script_Rx as Rxfunc
from virtualscanner.server.simulation.bloch import caller_script_blochsim as bsim

UPLOAD_FOLDER = './src/coms/coms_ui/static/user_uploads'
ALLOWED_EXTENSIONS = {'seq', 'jpg'}

# Define the location of template and static folders
# template_dir = os.path.abspath('../templates')
# static_dir=os.path.abspath('../static')

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

users = []
n_acqs = 0

app.secret_key = 'Session_key'


@app.route('/', methods=['POST',
                         'GET'])  # This needs to point to the login screen and then we can use the register link seprately
def log_in():
    session.clear()
    session['acq_out_axial'] = []
    session['acq_out_sagittal'] = []
    session['acq_out_coronal'] = []
    if request.method == 'POST':
        # users.append(request.form['user-name'])
        # session['username'] = users[-1]
        session['username'] = request.form['user-name']

        if request.form['mode'] == "Standard":
            return redirect("register")
        else:
            return redirect("recon")
    else:
        if 'username' in session and session['username'] in users:

            return render_template("log_in.html")
        else:

            return render_template("log_in.html")


@app.route('/register', methods=['POST',
                                 'GET'])  # This needs to point to the login screen and then we can use the register link seprately
def on_register():
    """

         Parameters
        ----------
           void

        Returns
        -------
           void (status in debug mode if required)

        Performs
        --------
            Renders the registration html page on the web
    """

    # serve register template
    if (session['username'] == ""):

        return redirect('')
    else:
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
    return render_template('register.html', success=session['reg_success'], payload=session['reg_payload'])


@app.route('/acquire', methods=['POST', 'GET'])
def on_acq():
    if 'acq' in session:

        return render_template('acquire.html', success=session['acq'], axial=session['acq_out_axial'],
                               sagittal=session['acq_out_sagittal'], coronal=session['acq_out_coronal'],
                               payload=session['acq_payload'])
    else:
        return render_template('acquire.html')


@app.route('/analyze', methods=['POST', 'GET'])
def on_analyze():
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


@app.route('/ana_load_success')
def on_ana_load_success():
    return render_template('analyze.html', load_success=session['ana_load'], payload1=session['ana_payload1'])


@app.route('/tx', methods=['POST', 'GET'])
def on_tx():
    if 'tx' in session:
        return render_template('tx.html', success=session['tx'], payload=session['tx_payload'])
    else:
        return render_template('tx.html')


@app.route('/rx', methods=['POST', 'GET'])
def on_rx():
    if 'rx' in session:
        return render_template('rx.html', success=session['rx'], payload=session['rx_payload'])
    else:
        return render_template('rx.html')


@app.route('/recon')
def on_recon():
    if 'recon' in session:
        return render_template('recon.html', success=session['recon'], payload=session['recon_payload'])
    else:
        return render_template('recon.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/receiver', methods=['POST', 'GET'])
def worker():
    """

            Parameters
            ----------
                   void

             Returns
            -------
               payload

            Performs
            --------
                rx payload from the client
                TODO: invokes payload
            """
    # read payload and convert it to dictionary

    if request.method == 'POST':
        payload = request.form.to_dict()

        # Registration
        if request.form['formName'] == 'reg':

            print(payload)
            session['reg_success'] = 1
            session['reg_payload'] = payload

            del payload['formName']
            # Right now only doing metric system.
            del payload['weight2']
            del payload['height2']
            del payload['measuresystem']

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
            print(payload)
            return redirect('register')
        # ACQUIRE
        elif request.form['formName'] == 'acq':

            session['acq'] = 0

            if (("patid" in session) == False):  # Please register first
                return redirect('register')

            pat_id = session['patid']
            query_dict = {
                "patid": pat_id,
            }

            rows = reg.reuse(query_dict)
            print(rows)

            # session['acq'] = 0
            session['acq_payload'] = payload

            progress = bsim.run_blochsim(seqinfo=payload, phtinfo=rows[0][0],
                                         pat_id=pat_id)  # phtinfo just needs to be 1 string
            sim_result_path = './src/coms/coms_ui/static/acq/outputs/' + session['patid']

            while (os.path.isdir(sim_result_path) is False):
                pass

            if progress == 1:
                session['acq'] = 1
                im_path_from_template = '../static/acq/outputs/' + session['patid']

                imgpaths = os.listdir(sim_result_path)
                complete_path = [im_path_from_template + '/' + iname for iname in imgpaths]
                Z_acq = []
                X_acq = []
                Y_acq = []
                for indx in range(len(complete_path)):

                    pos = complete_path[indx].find('_', 30, ) + 1

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

                if (("patid" in session) == False):  # Please register first
                    return redirect('register')

                if payload['original-data-opt'] == 'T1':
                    folder_path = './src/coms/coms_ui/static/ana/inputs/T1_original_data'
                elif payload['original-data-opt'] == 'T2':
                    folder_path = './src/coms/coms_ui/static/ana/inputs/T2_original_data'

                filenames_in_path = os.listdir(folder_path)
                original_data_path = ['./static/ana/inputs/' + payload['original-data-opt'] + '_original_data/' + iname
                                      for iname in filenames_in_path]

                payload['data-path'] = original_data_path

                session['ana_payload1'] = payload


            elif 'map-form' in payload:

                session['ana_map'] = 1

                if payload['TI'] == "":
                    server_od_path = './src/server/ana/inputs/T2_orig_data'
                    map_name, dicom_path = T2_mapping.main(server_od_path, payload['TR'], payload['TE'],
                                                           session['patid'])
                else:
                    server_od_path = './src/server/ana/inputs/T1_orig_data'
                    map_name, dicom_path = T1_mapping.main(server_od_path, payload['TR'], payload['TE'], payload['TI'],
                                                           session['patid'])

                # payload['map_path'] = '../static/ana/outputs/292/T1_map20190430142214.png'
                payload['dicom_path'] = dicom_path
                payload['map_path'] = '../static/ana/outputs/' + session['patid'] + '/' + map_name
                session['ana_payload2'] = payload


            elif 'roi-form' in payload:

                # payload['map-type'] = 'T1'

                session['ana_roi'] = 1
                dicom_map_path = session['ana_payload2']['dicom_path']
                if 'T1' in dicom_map_path:
                    payload['map-type'] = 'T1'
                elif 'T2' in dicom_map_path:
                    payload['map-type'] = 'T2'

                roi_result_filename = ROI_analysis.main(dicom_map_path, payload['map-type'], payload['map-size'],
                                                        payload['map-FOV'], session['patid'])

                roi_result_path = '../static/ana/outputs/' + session['patid'] + '/' + roi_result_filename

                # roi_result_path = '../static/ana/outputs/292/map_with_ROI20190430142228.png'
                payload['roi_path'] = roi_result_path
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

                timestamp = time.strftime("%Y%m%d%H%M%S")
                filename = filename[:-4] + timestamp + '.seq'

                os.rename(upload_path, "./src/server/rf/tx/SAR_calc/" + filename)

                output = SAR_calc_main.payload_process(filename)

                session['tx'] = 1
                output['plot_path'] = '../static/rf/tx/SAR/' + output['filename']
                session['tx_payload'] = output
                return redirect('tx')
        # rx
        elif request.form['formName'] == 'rx':
            print(payload)
            signals_path, recon_path, orig_im_path = Rxfunc.run_Rx_sim(payload)
            payload['signals_path'] = '..' + signals_path[18:]
            payload['recon_path'] = '..' + recon_path[18:]

            session['rx'] = 1
            session['rx_payload'] = payload
            return redirect('rx')

        # recon
        elif request.form['formName'] == 'recon':
            file = request.files['file']
            filename = secure_filename(file.filename)

            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_path)

            out_path_form_template = '../static/recon/outputs/'

            if payload['DL-type'] == "GT":
                out1, out2, out3 = main(input_path, payload['DL-type'])
                payload['output'] = [out_path_form_template + out1, out_path_form_template + out2,
                                     out_path_form_template + out3]
            else:
                out1, out2 = main(input_path, payload['DL-type'])
                payload['output'] = [out_path_form_template + out1, out_path_form_template + out2]

            session['recon'] = 1
            session['recon_payload'] = payload

            return redirect('recon')

    """
    payload = request.data
    payload = json.loads(payload.decode('utf8'))

    formName = payload.get('formName')
    # Do registration and save to database
    if formName == 'reg':
        if request.method == 'POST':
            session['reg_success'] = 1
            session['reg_payload'] = payload

        del payload['formName']

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

    """
    """
    if formName == 'acq':

        print("Arrived")
        return redirect('register')
        
        pat_id = session['patid']
        query_dict = {
            "patid": pat_id,
        }

        rows = reg.reuse(query_dict)
        print(rows)

        session['acq'] = 0

        print(session)


        #time.sleep(1)

        progress = bsim.run_blochsim(seqinfo=payload, phtinfo=rows[0][0],pat_id=pat_id)  # phtinfo just needs to be 1 string

        if request.method == 'POST' and progress == 1:
            session['acq'] = 1
            sim_result_path = '../static/acq/outputs/' + '1030'
            session['acq_output'] = sim_result_path + '/IM_GRE_20190425175047_1.png'
            #imgpaths = os.listdir(sim_result_path)
            #session['acq_output'] = [sim_result_path + '/'+ iname for iname in imgpaths]
        print(session)
    
    result = ''
    return result
        """


def launch_virtualscanner():
    app.run(host='0.0.0.0', debug=True)


if __name__ == '__main__':
    # run!
    # app.run(host='0.0.0.0', debug=True)
    # app.run(host='0.0.0.0', debug=True)
    launch_virtualscanner()
