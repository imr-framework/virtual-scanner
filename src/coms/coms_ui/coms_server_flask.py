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
   Tx to client
   Rx from client

Unit Test app
-------------
     utest_coms_flask
Author: Sairam Geethanath , Modified by: Marina Manso Jimeno
Date: 03/22/2019
Version 0.0
Copyright of the Board of Trustees of  Columbia University in the City of New York
"""

if __name__ == '__main__':
    import os
    import sys

    script_path = os.path.abspath(__file__)
    SEARCH_PATH = script_path[:script_path.index('Virtual-Scanner') + len('Virtual-Scanner') + 1]
    sys.path.insert(0, SEARCH_PATH)

import json
import time


from flask import Flask, render_template, request, redirect, session

import src.server.registration.register as reg
from src.server.simulation.bloch import caller_script_blochsim as bsim
from src.server.ana import T1_mapping as T1_mapping
from src.server.ana import T2_mapping as T2_mapping
from src.server.ana import ROI_analysis as ROI_analysis

# Define the location of template and static folders
# template_dir = os.path.abspath('../templates')
# static_dir=os.path.abspath('../static')

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True



users = []
app.secret_key = 'Session_key'


@app.route('/', methods=['POST','GET'])  # This needs to point to the login screen and then we can use the register link seprately
def log_in():
    session.clear()
    if request.method == 'POST':
        #users.append(request.form['user-name'])
        #session['username'] = users[-1]
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


@app.route('/register',methods =['POST','GET'])  # This needs to point to the login screen and then we can use the register link seprately
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

@app.route('/register_success',methods=['POST','GET'])
def on_register_success():
    return render_template('register.html', success=session['reg_success'], payload=session['reg_payload'])


@app.route('/acquire',methods=['POST','GET'])
def on_acq():

    return render_template('acquire.html')

@app.route('/acquire_success', methods=['POST','GET'])
def on_acquire_success():

    if session['acq'] == 1:

        return render_template('acquire.html', success=session['acq'],output_im=session['acq_output'],payload=session['acq_payload'])
    else:

        return redirect('acquire')


@app.route('/analyze')
def on_analyze():
    if 'ana_load' in session:
        if 'ana_map' in session:
            if 'ana_roi' in session:
                return render_template('analyze.html', roi_path=session['roi_result'])
            else:
                return render_template('analyze.html', map_success=session['ana_map'], load_success=session['ana_load'], payload12=session['ana_payload1'], payload2 = session['ana_payload2'])
        else:
            return render_template('analyze.html',load_success=session['ana_load'],payload1=session['ana_payload1'])

    else:
        return render_template('analyze.html')



@app.route('/recon')
def on_recon():
    return render_template('recon.html')


@app.route('/receiver', methods=['POST','GET'])
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
                Rx payload from the client
                TODO: invokes payload
            """
    # read payload and convert it to dictionary


    if request.method == 'POST':
        payload = request.form.to_dict()


        #Registration
        if request.form['formName'] == 'reg':


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

        #ACQUIRE
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

            #session['acq'] = 0
            session['acq_payload'] = payload

            progress = bsim.run_blochsim(seqinfo=payload, phtinfo=rows[0][0],pat_id=pat_id)  # phtinfo just needs to be 1 string
            sim_result_path = './src/coms/coms_ui/static/acq/outputs/' + session['patid']


            while (os.path.isdir(sim_result_path) is False):
                pass

            if progress == 1 :
                session['acq'] = 1
                im_path_from_template = '../static/acq/outputs/' + session['patid']

                imgpaths = os.listdir(sim_result_path)
                complete_path = [im_path_from_template + '/'+ iname for iname in imgpaths]
                #TODO: get all the available images in the folder
                session['acq_output'] = complete_path[0]

                return redirect('acquire_success')

        elif request.form['formName'] == 'ana':

            if 'original-data-opt' in payload:

                session['ana_load'] = 1



                if (("patid" in session) == False): #Please register first
                    return redirect('register')

                if payload['original-data-opt'] == 'T1' :
                    folder_path = './src/coms/coms_ui/static/ana/inputs/T1_original_data'
                elif payload['original-data-opt'] == 'T2' :
                    folder_path = './src/coms/coms_ui/static/ana/inputs/T2_original_data'

                filenames_in_path = os.listdir(folder_path)
                original_data_path = ['./static/ana/inputs/'+ payload['original-data-opt'] + '_original_data/'+ iname for iname in filenames_in_path]

                payload['data-path'] = original_data_path

                session['ana_payload1'] = payload

            elif 'map-form' in payload:


                session['ana_map'] = 1

                if payload['TI'] == "":
                    server_od_path = './src/server/ana/inputs/T2_orig_data'
                    map_name = T2_mapping.main(server_od_path, payload['TR'], payload['TE'], session['patid'])
                else:
                    server_od_path = './src/server/ana/inputs/T1_orig_data'
                    map_name = T1_mapping.main(server_od_path,payload['TR'],payload['TE'],payload['TI'],session['patid'])



                payload['map_path'] = '../static/ana/outputs/' + session['patid'] + '/' + map_name
                session['ana_payload2'] = payload


            elif 'roi-form' in payload:

                print(payload)
                payload['map-type'] = 'T1'

                session['ana_roi'] = 1

                if payload['map-type'] == 'T1':

                    dicom_map_path = './src/server/ana/outputs/T1_map'


                else:

                    dicom_map_path = './src/server/ana/outputs/T2_map'

                roi_result_filename = ROI_analysis.main(dicom_map_path, payload['map-type'], payload['map-size'], payload['map-FOV'], session['patid'])

                roi_result_path='../static/ana/outputs/' + session['patid']+'/' + roi_result_filename
                print (roi_result_path)
                session['roi_result']= roi_result_path
                print(session)

            return redirect('analyze')




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

if __name__ == '__main__':
    # run!
    # app.run(host='0.0.0.0', debug=True)
    app.run(host='0.0.0.0', debug=True)
