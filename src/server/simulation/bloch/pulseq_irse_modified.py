"""
This is starter code to demonstrate a working example of an Inversion Recovery Spin Echo (IRSE) sequence
     in a pure Python implementation.
"""
from math import pi

from pulseq.core.Sequence.sequence import Sequence
from pulseq.core.calc_duration import calc_duration
from pulseq.core.make_adc import makeadc
from pulseq.core.make_delay import make_delay
from pulseq.core.make_sinc import make_sinc_pulse
from pulseq.core.make_trap import make_trapezoid
from pulseq.core.opts import Opts

kwargs_for_opts = {"max_grad": 33, "grad_unit": "mT/m", "max_slew": 100, "slew_unit": "T/m/s", "rf_dead_time": 10e-6,
                   "adc_dead_time": 10e-6}
system = Opts(kwargs_for_opts)
seq = Sequence(system)

# TODO is there a problem with phase encodes ???

fov = 32
Nx = 15
Ny = 15
slice_thickness = 32/15

#TI = [0.021, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
TI = [0.2]
TE, TR = 12e-3, 10

flip = 90 * pi / 180
kwargs_for_sinc = {"flip_angle": flip, "system": system, "duration": 2e-3, "slice_thickness": slice_thickness,
                   "apodization": 0.5, "time_bw_product": 4}
rf, gz = make_sinc_pulse(kwargs_for_sinc, 2)
# plt.plot(rf.t[0], rf.signal[0])
# plt.show()

delta_k = 1 / fov
kWidth = Nx * delta_k
readoutTime = system.grad_raster_time * Nx
kwargs_for_gx = {"channel": 'x', "system": system, "flat_area": kWidth, "flat_time": readoutTime}
gx = make_trapezoid(kwargs_for_gx)
kwargs_for_adc = {"num_samples": Nx, "system": system, "duration": gx.flat_time, "delay": gx.rise_time}
adc = makeadc(kwargs_for_adc)

kwargs_for_gxpre = {"channel": 'x', "system": system, "area": gx.area / 2, "duration": readoutTime / 2}
gx_pre = make_trapezoid(kwargs_for_gxpre)
kwargs_for_gz_reph = {"channel": 'z', "system": system, "area": -gz.area / 2, "duration": 2e-3}
gz_reph = make_trapezoid(kwargs_for_gz_reph)

flip = 180 * pi / 180
kwargs_for_sinc = {"flip_angle": flip, "system": system, "duration": 2e-3, "slice_thickness": slice_thickness,
                   "apodization": 0.5, "time_bw_product": 4}
rf180, gz180 = make_sinc_pulse(kwargs_for_sinc, 2)


delayTE1 = TE / 2 - calc_duration(gz_reph) - calc_duration(rf) - calc_duration(rf180) / 2
delayTE2 = TE / 2 - calc_duration(gx) / 2 - calc_duration(rf180) / 2
delayTE3 = TR - TE - calc_duration(gx) # TODO seems wrong? check ACTUAL "TR" in plot
delay1 = make_delay(delayTE1)
delay2 = make_delay(delayTE2)
delay3 = make_delay(delayTE3)



for inv in range(len(TI)):
    for i in range(Ny):
        # Inversion Recovery part
        seq.add_block(rf180)  # Add inversion pulse # Non-selective at the moment, could be extended to make this selective/adiabatic
        seq.add_block(make_delay(TI[inv]))  # Inversion time delay
        # Spin echo part
        seq.add_block(rf, gz) # 90-deg pulse

        kwargs_for_gy_pre = {"channel": 'y', "system": system, "area": -(Ny / 2 - i) * delta_k, "duration": readoutTime / 2}
        gy_pre = make_trapezoid(kwargs_for_gy_pre) # Phase encoding gradient
        seq.add_block(gx_pre, gy_pre, gz_reph) # Add a combination of ro rewinder, phase encoding, and slice refocusing

        seq.add_block(delay1) # Delay 1: until 180-deg pulse

        seq.add_block(rf180, gz180) # 180 deg pulse for SE

        seq.add_block(delay2)  # Delay 2: until readout

        seq.add_block(gx, adc) # Readout!

        seq.add_block(delay3) # Delay 3: until next inversion pulse
        print("another added")

# Display 2 TR
seq.plot(time_range=(0, 3*TR))

# Display entire plot
seq.plot()

# The .seq file will be available inside the /gpi/<user>/imr_framework folder
seq.write("irse_python_forsim_15_fov32_rev.seq")
