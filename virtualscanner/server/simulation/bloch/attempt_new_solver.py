# Test the new NumSolverSpinGroup class
from bloch.spingroup_ps_wsolver import NumSolverSpinGroup
from rf_sim.rf_simulations import make_rf_shapes
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Define problem

    tmodel, pulse_shape = make_rf_shapes(pulse_type="sinc",flip_angle=90,nzc=6, bw_rf=5e3, dt=1e-6)

    # No relaxation
    sg = NumSolverSpinGroup(loc=(0,0,0), pdt1t2=(1,0,0), df=0)
    # Zero gradient
    dt = 1e-6
    grads_shape = np.zeros((3, len(pulse_shape)))
    results = sg.apply_rf_store(pulse_shape, grads_shape, dt)

    # find signal from results.y
    mx = results.y[0,:]
    my = results.y[1,:]
    mz = results.y[2,:]

    print(np.shape(results.y))
    print(np.shape(results.t))

    plt.figure(1)
    plt.plot(results.t, mx,label="My")
    plt.plot(results.t, my,label="Mx")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("Mx, My")
 #   plt.show()