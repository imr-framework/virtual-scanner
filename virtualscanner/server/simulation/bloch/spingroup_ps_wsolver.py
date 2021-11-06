# Spingroup with solver
from bloch.spingroup_ps import SpinGroup
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d



GAMMA = 2*42.58e6 * np.pi
GAMMA_BAR = 42.58e6

class NumSolverSpinGroup(SpinGroup):
    # TODO package the funtions to generate a final function that only takes in t and M and returns dM/dt

    @staticmethod
    def interpolate_waveforms(grads_shape, pulse_shape, dt):
        # Helper function to generate continuous waveforms
        gx_func = interp1d(x=dt*np.arange(len(pulse_shape)), y=grads_shape[0,:])
        gy_func = interp1d(x=dt*np.arange(len(pulse_shape)), y=grads_shape[1,:])
        gz_func = interp1d(x=dt*np.arange(len(pulse_shape)), y=grads_shape[2,:])
        pulse_real_func = interp1d(x=dt*np.arange(len(pulse_shape)), y=np.real(pulse_shape))
        pulse_imag_func = interp1d(x=dt*np.arange(len(pulse_shape)), y=np.imag(pulse_shape))

        return gx_func, gy_func, gz_func, pulse_real_func, pulse_imag_func

    # TODO make this return the input diffEQ to solver
    def get_bloch_eqn(self, grads_shape, pulse_shape, dt):
        x, y, z = self.loc
        gx_func, gy_func, gz_func, pulse_real_func, pulse_imag_func = self.interpolate_waveforms(grads_shape, pulse_shape, dt)

        T1_inv = 1 / self.T1 if self.T1 > 0 else 0
        T2_inv = 1 / self.T2 if self.T2 > 0 else 0
        dB = self.df / GAMMA_BAR

        def bloch_eqn(t,m):
            gx, gy, gz = gx_func(t), gy_func(t), gz_func(t)
            B1x = pulse_real_func(t)
            B1y = pulse_imag_func(t)
            glocp = gx*x + gy*y + gz*z
            A = np.array([[-T2_inv, GAMMA * (dB + glocp), -GAMMA * B1y],
                          [-GAMMA * (dB + glocp), -T2_inv, GAMMA * B1x],
                          [GAMMA * B1y, -GAMMA * B1x, -T1_inv]])
            return A @ m + np.array([[0], [0], [T1_inv]])


        return bloch_eqn


    # Override RF method!
    def apply_rf_store(self, pulse_shape, grads_shape, dt):
        m = np.squeeze(self.m)

        ####
        m_signal = np.zeros(len(pulse_shape), dtype=complex)
        magnetizations = np.zeros((3, len(pulse_shape) + 1))
        magnetizations[:, 0] = np.squeeze(m)
        ####

        # Set correct arguments to ivp solver ...
        results = solve_ivp(fun=self.get_bloch_eqn(grads_shape,pulse_shape,dt), t_span=[0,len(pulse_shape)*dt-dt],
                            y0=m, t_eval=dt*np.arange(len(pulse_shape)), vectorized=True)

        print(results)

        return results





