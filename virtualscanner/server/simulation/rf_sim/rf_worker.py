import numpy as np
import plotly.graph_objects as go
import json
import plotly
from plotly.subplots import make_subplots
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_block_pulse import make_block_pulse
from pypulseq.make_gauss_pulse import make_gauss_pulse
from pypulseq.opts import Opts
from virtualscanner.server.simulation.rf_sim.rf_simulations import simulate_rf

GAMMA_BAR = 42.58e6

def rf_display_worker(info):
    rf, t, _ = get_pulse(info)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=t, y=np.angle(rf), mode='lines', name="RF phase",
                   line=dict(color='red', width=1)),secondary_y=True
    )
    fig.add_trace(
        go.Scatter(x=t, y=np.absolute(rf), mode='lines', name='RF magnitude',
                   line=dict(color='blue', width=2)),secondary_y=False
    )

    fig.update_layout(title_text="RF pulse",showlegend=False)
    fig.update_yaxes(title_text="RF magnitude",secondary_y=False)
    fig.update_yaxes(title_text="RF phase",secondary_y=True)

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

def rf_simulate_worker(info):
    rf, t, thk_bw = get_pulse(info)

    # Call simulation function
    bw_spins = info['spin_bw']
    num_spins = info['spin_num']
    print('dt: ', t[1]-t[0])
    print('rf: ', rf)
    signals, ms = simulate_rf(bw_spins= bw_spins,
                                 n_spins=num_spins, pdt1t2=(1, info['spin_t1']*1e-3, info['spin_t2']*1e-3),# TODO add in PD/T1/T2 variation
                                 flip_angle=90,
                                 dt=t[1]-t[0],
                                 solver="RK45",
                                 pulse_type='custom',
                                 pulse_shape=rf / GAMMA_BAR, display=False)
    print('rf simulated!')
    fmodel = np.linspace(-0.5 * bw_spins, 0.5 * bw_spins, len(signals))

    # First figure
    fig = make_subplots(rows=3, cols=1)
    pf = 4
    fig.add_trace(go.Scatter(x=[-thk_bw / 2, thk_bw / 2, thk_bw / 2, -thk_bw / 2],
                             y=pf*np.array([-1,-1,1,1]), fill="toself", line=dict(color="rgba(0,0,0,0)"),
                             fillcolor="LightSteelBlue"), row=1, col=1)

    fig.add_trace(go.Scatter(x=[-thk_bw / 2, thk_bw / 2, thk_bw / 2, -thk_bw / 2],
                             y=pf*np.array([-1,-1,1,1]), fill="toself", line=dict(color="rgba(0,0,0,0)"),
                             fillcolor="LightSteelBlue"), row=2, col=1)


    fig.add_trace(go.Scatter(x=[-thk_bw / 2, thk_bw / 2, thk_bw / 2, -thk_bw / 2],
                             y=pf*np.array([-1,-1,1,1]), fill="toself", line=dict(color="rgba(0,0,0,0)"),
                             fillcolor="LightSteelBlue"), row=3, col=1)

    fig.add_trace(go.Scatter(x=fmodel, y=np.absolute(signals[:, -1]), mode="lines", line=dict(width=2, color="navy")),
                   row=1, col=1)
    fig.add_trace(go.Scatter(x=fmodel, y=np.angle(signals[:, -1]), mode="lines", line=dict(width=2, color="maroon")),
                   row=2, col=1)
    fig.add_trace(go.Scatter(x=fmodel, y=np.squeeze(ms[:, 2, -1]), mode="lines", line=dict(width=2, color="teal")),
                   row=3, col=1)

    fig.update_layout(title_text="RF slice profile",showlegend=False)
    fig.update_yaxes(title_text="Mxy magnitude",range=[-0.1,1.1],row=1,col=1)
    fig.update_yaxes(title_text="Mxy phase",range=[-np.pi,np.pi],row=2,col=1)
    fig.update_yaxes(title_text="Mz",range=[-1.1,1.1],row=3,col=1)

    fig.update_xaxes(range=[fmodel[0],fmodel[-1]],row=1,col=1)
    fig.update_xaxes(range=[fmodel[0],fmodel[-1]],row=2,col=1)
    fig.update_xaxes(range=[fmodel[0],fmodel[-1]],row=3,col=1)

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Second figure
    # Display up to 16 spins
    # Retrieve data
    mxs = np.squeeze(ms[:, 0, :])
    mys = np.squeeze(ms[:, 1, :])
    mzs = np.squeeze(ms[:, 2, :])

    if mxs.shape[0] > 16:
        # Get 16 equally spaced samples
        indices = np.floor(np.arange(16) * (ms.shape[0]/16)).astype(int)
        mxs = mxs[indices,:]
        mys = mys[indices,:]
        mzs = mzs[indices,:]
        fmodel = fmodel[indices]

    mxy_mag = np.absolute(mxs + 1j*mys)
    mxy_phase = np.angle(mxs + 1j*mys)

    # Make figure and put data on
    fig = go.Figure()
    tmodel = t
    for b in range(mxs.shape[0]):
        fig.add_trace(go.Scatter(x=tmodel*1e3, y=mxs[b,:],mode="lines",name=f"{round(fmodel[b],2)}Hz",
                                 line=dict(color=get_color_mapped(b/mxs.shape[0])))) # Mx only? TODO add the other two.

    fig.update_layout(title_text="Spin evolution",
                      updatemenus=[
                          dict(
                              type="buttons",
                              direction="down",
                              pad={"r": 30},
                              showactive=True,
                              x=0,
                              y=1,
                              xanchor="right",
                              yanchor="top",
                              buttons=list([
                                  dict(
                                      args=[
                                          {'y': mxs, "title_text":"Mx"}
                                      ],
                                      label="Mx",
                                      method="update"
                                  ),
                                  dict(
                                      args=[
                                          {'y': mys, 'title_text':"My"}
                                      ],
                                      label="My",
                                      method="update"
                                  ),
                                  dict(
                                      args=[
                                          {'y': mzs, }
                                      ],
                                      label="Mz",
                                      method="update"
                                  ),
                                  dict(
                                      args=[
                                          {'y': mxy_mag, }
                                      ],
                                      label="|Mxy|",
                                      method="update"
                                  ),
                                  dict(
                                      args=[
                                          {'y': mxy_phase, }
                                      ],
                                      label="angle(Mxy)",
                                      method="update"
                                  )

                              ])

                          )
                      ])
    fig.update_xaxes(title_text="Time (ms)")



    graphJSON2 = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON, graphJSON2




def get_pulse(info):
    system = set_June_system_limits()
    if info['pulse_type'] == 'sinc90':
        rf, gz, gz_ref = make_sinc_pulse(flip_angle=np.pi/2, system=system, duration=1e-3, slice_thickness=5e-3,
                                         apodization=0.5, time_bw_product=4, phase_offset=0,
                                         return_gz=True)
        thk_bw = (5e-3) * gz.amplitude # Hz
        print('thkbw:',thk_bw)

    elif info['pulse_type'] == 'sinc180':
        rf, gz, gz_ref = make_sinc_pulse(flip_angle=np.pi, system=system, duration=1e-3, slice_thickness=5e-3,
                                         apodization=0.5, time_bw_product=4, phase_offset=0,
                                         return_gz=True)
        thk_bw = (5e-3) * gz.amplitude # Hz


    elif info['pulse_type'] == 'blk90':
        rf, gz = make_block_pulse(flip_angle=np.pi/2, system=system, duration=1e-3, slice_thickness=5e-3, return_gz=True)
        thk_bw = (5e-3) * gz.amplitude

    elif info['pulse_type'] == 'custom':
        print(info['rf_shape'])
        if info['rf_shape'] == 'sinc':
            rf, gz, _ = make_sinc_pulse(flip_angle=info['rf_fa']*np.pi/180, system=system, duration=info['rf_dur']*1e-3,
                                     slice_thickness=info['rf_thk']*1e-3, freq_offset=info['rf_df'],
                                     phase_offset=info['rf_dphi']*np.pi/180, time_bw_product=info['rf_tbw'],return_gz=True)
            thk_bw = info['rf_thk'] * 1e-3 * gz.amplitude

        elif info['rf_shape'] == 'block':
            rf, gz = make_block_pulse(flip_angle=info['rf_fa'] * np.pi / 180, system=system,
                                     duration=info['rf_dur'] * 1e-3,
                                     slice_thickness=info['rf_thk'] * 1e-3, freq_offset=info['rf_df'],
                                     phase_offset=info['rf_dphi'] * np.pi / 180, time_bw_product=info['rf_tbw'],return_gz=True)
            thk_bw = info['rf_thk'] * 1e-3 * gz.amplitude
        elif info['rf_shape'] == 'gauss':
            rf, gz, _ = make_gauss_pulse(flip_angle=info['rf_fa'] * np.pi / 180, system=system,
                                     duration=info['rf_dur'] * 1e-3,
                                     slice_thickness=info['rf_thk'] * 1e-3, freq_offset=info['rf_df'],
                                     phase_offset=info['rf_dphi'] * np.pi / 180, time_bw_product=info['rf_tbw'],return_gz=True)
            thk_bw = info['rf_thk'] * 1e-3 * gz.amplitude

    elif info['pulse_type'] == 'sigpy':
        raise NotImplementedError

    return rf.signal, rf.t, thk_bw

# TODO  convert to just code and return!
def generate_rf_code(info):
    system = set_June_system_limits()
    code = ""
    if info['pulse_type'] == 'sinc90':
        code += f"rf, gz, gz_ref = make_sinc_pulse(flip_angle=np.pi/2, system=system, duration=1e-3, slice_thickness=5e-3,"
        code += "apodization=0.5, time_bw_product=4, phase_offset=0,return_gz=True)"


    elif info['pulse_type'] == 'sinc180':
        code += "rf, gz, gz_ref = make_sinc_pulse(flip_angle=np.pi, system=system, duration=1e-3, slice_thickness=5e-3,"
        code += "apodization=0.5, time_bw_product=4, phase_offset=0,return_gz=True)"

    elif info['pulse_type'] == 'blk90':
        code += "rf, gz = make_block_pulse(flip_angle=np.pi/2, system=system, duration=1e-3, slice_thickness=5e-3, return_gz=True)"

    elif info['pulse_type'] == 'custom':
        if info['rf_shape'] == 'sinc':
            # Declare values
            code += f"rf, gz, gzref = make_sinc_pulse(flip_angle={info['rf_fa']*np.pi/180}, system=system, duration={info['rf_dur']*1e-3},"
            code += f"slice_thickness={info['rf_thk']*1e-3}, freq_offset={info['rf_df']},"
            code += f"phase_offset={info['rf_dphi']*np.pi/180}, time_bw_product={info['rf_tbw']},return_gz=True)"

        elif info['rf_shape'] == 'block':
            code += f"rf, gz = make_block_pulse(flip_angle={info['rf_fa'] * np.pi / 180}, system=system,"
            code += f"duration={info['rf_dur'] * 1e-3}, slice_thickness={info['rf_thk'] * 1e-3}, freq_offset={info['rf_df']},"
            code += f"phase_offset={info['rf_dphi'] * np.pi / 180}, time_bw_product={info['rf_tbw']},return_gz=True)"
        elif info['rf_shape'] == 'gauss':
            code += f"rf, gz, _ = make_gauss_pulse(flip_angle={info['rf_fa']*np.pi/180}, system=system,"
            code += f"duration={info['rf_dur']*1e-3}, slice_thickness={info['rf_thk']*1e-3}, freq_offset={info['rf_df']},"
            code += f"phase_offset={info['rf_dphi']*np.pi/180}, time_bw_product={info['rf_tbw']},return_gz=True)"

    return code

# system: use June defaults for now
def set_June_system_limits():
    system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130,
                      slew_unit='T/m/s', rf_ringdown_time=100e-6, # changed from 30e-6
                      rf_dead_time=100e-6, adc_dead_time=20e-6)
    return system



def get_color_mapped(scale=0):
    orange = np.array([255,165,0])
    purple = np.array([128,0,128])

    colors = purple + scale*(orange - purple)
    return f'rgb{tuple(colors)}'