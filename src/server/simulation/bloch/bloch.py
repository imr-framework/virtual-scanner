import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import copy

"""
Essential components for bloch simulation
Gehua Tong
March 2019
"""


# Gradient class
class Gradient:
    def __init__(self,area,duration):
        self._area = area
        self._duration = duration

    def get_area(self):
        return self._area

    def get_duration(self):
        return self._duration

    def get_amplitude(self):
        if self._duration != 0:
            return self._area/self._duration
        else:
            return 0

    def scale(self,s):
        self._area = self._area*s

    def __str__(self):
        return "Gen. gradient with area " + str(self._area) + " and timing " + str(self._duration)

# Trapezoidal gradient
class TrapzGradient(Gradient):
    def __init__(self,rise_time,flat_time,fall_time,amplitude):
        my_area = amplitude*(flat_time + rise_time/2 + fall_time/2)
        super().__init__(area=my_area, duration=rise_time + flat_time + fall_time)
        self._rise_time = rise_time
        self._flat_time = flat_time
        self._fall_time = fall_time
        self._amplitude = amplitude

    def __str__(self):
        return "Gradient @ flat time " + str(self._flat_time) + " and amplitude " + str(self._amplitude)

    def get_amplitude(self):
        return self._amplitude

    def get_rise_time(self):
        return self._rise_time

    def get_flat_time(self):
        return self._flat_time

    def get_fall_time(self):
        return self._fall_time

    def plot(self):
        rt = self._rise_time()
        ft = self._flat_time()
        dur = self._duration()
        amp = self._amplitude()
        plt.plot([0,rt,rt+ft,dur],[0,amp,amp,0])
        plt.show()

    def scale(self, s):
        super().scale(s)
        self._amplitude = self._amplitude*s

# Time-resolved gradient
class VaryingGradient(Gradient):
    def __init__(self,shape_func,raster_time,interval):
        self._shape_func = shape_func
        self._interval = interval
        self._dt = raster_time
        self._timing = np.arange(interval[0],interval[1]+self._dt,self._dt)
        self._shape = shape_func(self._timing)

        my_area = np.zeros((3, np.shape(self._timing)[0]))
        for a in range(3):
            my_area[a] = np.trapz(self._shape[a],self._timing)

        super().__init__(area=my_area, duration=self._timing[-1]-self._timing[0])

    def get_shape_func(self):
        return self._shape_func

    def get_raster_time(self):
        return self._dt

    def get_interval(self):
        return self._interval

    def get_shape(self):
        return self._shape

    def get_timing(self):
        return self._timing

    def plot(self):
        plt.plot(self._timing,self._shape)
        plt.show()

    def scale(self,s):
        super().scale(s)
        self._shape = s*self._shape
        self._shape_func = s*self._shape_func

# RF pulses
class RFPulse:
    GAMMA = 42.58e6 * 2 * np.pi
    def __init__(self,shape,timing):
        self._shape = shape
        self._timing = timing
        self._duration = timing[-1] - timing[0]

    def get_shape(self):
        return self._shape

    def get_timing(self):
        return self._timing

    def get_duration(self):
        return self._duration

    def get_flip_angle(self):
        return np.trapz(self._shape,self._timing)*self.GAMMA




# TODO fix construction of sinc pulse and try plotting
class SincRFPulse(RFPulse):
    """
    Sinc-shaped RF pulse
    INPUTS:
        bandwidth (Hz), offset (Hz), num_zeros - # zero crossing,
        flip_angle (rad), raster_time (s), init_phase (rad)
    """
    def __init__(self,bandwidth,offset,num_zeros,flip_angle,raster_time=1e-6,init_phase=0):
        # Store variables: bw = bandwidth; nzc = number of zero crossings; flip = flip angle (rad)
        self._bw = bandwidth
        self._nzc = num_zeros
        self._fa = flip_angle
        self._dt = raster_time
        self._df = offset
        self._phi0 = init_phase

        dt = self._dt

        # Create pulse shape and store
        t_max = (1/self._bw)*self._nzc/2
        t_model = np.arange(-t_max,t_max+dt,dt)
        envelope = np.sinc(self._bw*t_model)

        alpha0 = np.trapz(envelope,t_model)*self.GAMMA
        amp = self._fa/alpha0

        t = t_model + t_max
        B1 = np.multiply(amp*envelope, np.exp(-1j*(self._phi0 + 2*np.pi*self._df*t)))

        super().__init__(shape=B1,timing=t)

    def get_offset(self):
        """
        Returns center frequency of pulse
        """
        return self._df

    def get_bandwidth(self):
        """
        Returns frequency range of pulse
        """
        return self._bw

    def get_raster_time(self):
        return self._dt

    def get_flip_angle(self):
        return self._fa

    def plot(self):
        """
        Plots rf pulse using matplotlib
        """
        plt.plot(self._timing,self._shape)
        plt.show()

    def add_offset(self,df):
        """
        Adds an offset to the RF pulse.
            INPUT: df (Hz)
        """
        # Offsets center frequency of RF pulse
        self._df += df
        self._shape = np.multiply(self._shape, np.exp(-1j*np.pi*2*df*self._timing))



class ADC:
    """
    ADC sampling scheme
    """
    def __init__(self,num_samples,dwell_time,delay=0):
        self._n = num_samples
        self._dt = dwell_time
        self._delay = delay

    def get_num_samples(self):
        return self._n

    def get_dwell_time(self):
        return self._dt

    def get_delay(self):
        return self._delay

    def get_bandwidth(self):
        return 1/self._dt

    def get_duration(self):
        return self._n*self._dt



class SpinGroup:
    """
    Basic unit of simulation
    Contains location, proton density, T1, T2, and off-resonance info
    """

    GAMMA = 2 * np.pi * 42.58e6
    GAMMA_BAR = 42.58e6

    def __init__(self, loc=(0,0,0), params=(1,0,0), df=0):
        """
        Define a spin group (with defined T1,T2,PD,location and frequency offset)
        loc: physical location w/ respect to isocenter (m)
        params: (PD,T1,T2) (a.u., sec, sec)
        df: off-resonance (Hz)
        """

        self._m = np.array([[0], [0], [1]])
        self._loc = loc
        self._params = params
        self._df = df

    def get_loc(self):
        return self._loc

    def get_df(self):
        return self._df

    def get_m(self):
        """
        Returns magnetization vector, scaled by PD
        """
        return self._params[0]*self._m

    def get_m_signal(self):
        """
        Returns complex rep. of Mxy, scaled by PD
        """
        return self._params[0]*(self._m[0] + 1j * self._m[1])

    def fpwg(self, grads):
        """
        free precession with gradients for the duration of the longest gradient in the list
        INPUT: grads: list of Gradient objects [g1, g2, g3, ...]

        duration of precession  = duration of longest gradient
        amount of phase change = all 3D gradients combined

        """
        dur = 0
        grad_area = np.array([0,0,0]).astype(float)
        for g in grads:
            dur = np.maximum(dur, g.get_duration())
            grad_area += g.get_area()
        phi = self.GAMMA*np.sum(np.multiply(self._loc, grad_area))+2*np.pi*self._df*dur
        C = np.cos(phi)
        S = np.sin(phi)
        T1 = self._params[1]
        T2 = self._params[2]
        if T1 == 0:
            E1 = 1
        else:
            E1 = np.exp(-dur/T1)

        if T2 == 0:
            E2 = 1
        else:
            E2 = np.exp(-dur/T2)
        # Clockwise rotation by +phi (i.e. CCW rotation by -phi)
        A = np.array([[E2*C, E2*S, 0],
                      [-E2*S, E2*C, 0],
                      [0, 0, E1]])
        self._m = A@self._m + np.array([[0],[0],[1 - E1]])



    def delay(self, t):
        """
        Applies a time passage to the spin group
        i.e. T1, T2 relaxation with no gradients
            INPUTS: t (s) - delay interval
        """
        zero_grad = Gradient(0,t)
        self.fpwg([zero_grad])

    def apply_rf(self, pulse, grad):
        """
        Applies a RF pulse to the spin group
        Simulation method: hard-pulse approximation across small time intervals
        With a trapezoid gradient, rf is applied only on the flat part
        """
        loc = self.get_loc()
        b1 = pulse.get_shape()
        rf_time = pulse.get_timing()
        dt = pulse.get_raster_time()

        if isinstance(grad,TrapzGradient):
            # Approximate rf + gradient as dt-wise discrete rotations
            flat_g = grad.get_amplitude()
            # ramp up
            self.fpwg([Gradient(0.5*grad.get_rise_time()*flat_g,grad.get_rise_time())])
            # flat part
            for k in range(np.shape(rf_time)[0]):
                bx = np.real(b1[k])
                by = np.imag(b1[k])
                bz = np.sum(np.multiply(flat_g,loc)) + self._df/self.GAMMA_BAR
                be = np.array([bx,by,bz])
                self._m = anyrot(self.GAMMA*be*dt)@self._m
            # ramp down
            self.fpwg([Gradient(0.5*grad.get_fall_time()*flat_g,grad.get_fall_time())])

        elif isinstance(grad,VaryingGradient):
            grad_shape = grad.get_shape()
            if np.shape(grad_shape) != np.shape(b1):
                raise ValueError("rf and gradient sequences are not the same length")
            for k in range(np.shape(rf_time)[0]):
                bx = np.real(b1[k])
                by = np.imag(b1[k])
                bz = np.sum(np.multiply(grad_shape[k],loc))+ self._df/self.GAMMA_BAR
                be = np.array([bx,by,bz])
                self._m = anyrot(self.GAMMA*be*dt)@self._m

        else:
            # generic gradient with area & duration - treat as perfect rect gradient
            flat_g = grad.get_amplitude()
            for k in range(np.shape(rf_time)[0]):
                bx = np.real(b1[k])
                by = np.imag(b1[k])
                bz = np.sum(np.multiply(flat_g, loc)) + self._df / self.GAMMA_BAR
                be = np.array([bx, by, bz])
                self._m = anyrot(self.GAMMA*be*dt)@self._m

    def flat_readout(self,grad_level,adc):
        n = adc.get_num_samples()
        signal = np.zeros(n,dtype=np.complex_)
        dt = adc.get_dwell_time()
        diff_grad = Gradient(area=grad_level*dt,duration=dt)
        for p in range(n):
            signal[p] = self.get_m_signal()
            self.fpwg([diff_grad])
        return signal

    def readout(self, grad, adc):
        dt = adc.get_dwell_time()
        n = adc.get_num_samples()
        delay = adc.get_delay()
        signal = np.zeros((1,n))
        adc_dur = adc.get_duration()
        # Apply gradient for each dt interval and record m
        if isinstance(grad,TrapzGradient):
            post_delay = grad.get_flat_time() - (delay + adc_dur)
            if post_delay < 0:
                raise ValueError("Delay + ADC sampling exceeds flat time of trapz gradient! :(")
            else:
                # calculate post-adc gradient time (already made sure it >= 0 given the conditions)
                # Ramp up
                self.fpwg([Gradient(0.5*grad.get_rise_time()*grad.get_amplitude(),grad.get_rise_time())])
                # Delay
                self.fpwg([Gradient(delay*grad.get_amplitude(),delay)])
                # Constant amplitude gradient readout
                signal = self.flat_readout(grad.get_amplitude(),adc)
                # Post-delay
                self.fpwg([Gradient(post_delay*grad.get_amplitude(),post_delay)])
                # Ramp down
                self.fpwg([Gradient(0.5*grad.get_fall_time()*grad.get_amplitude(),grad.get_fall_time())])

        elif isinstance(grad,VaryingGradient):
            # most general; use adc delay!
            # Check timing
            post_delay = grad.get_duration() - (delay + adc_dur)
            if post_delay < 0:
                raise ValueError("Delay + ADC sampling period is longer than gradient duration :(")
            ti = grad.get_timing()[0]
            tf = grad.get_timing()[-1]
            grad_shape = grad.get_shape()

            # delay before readout
            self.fpwg([Gradient(find_approx_area(grad.get_timing(),grad_shape,(ti,ti+delay)),delay)])

            for p in range(adc.get_n()):
                signal[p] = self.get_m_signal()
                diff_grad = Gradient(area=grad_shape[p]*dt, duration=dt)
                self.fpwg([diff_grad])

            # delay after readout
            self.fpwg([Gradient(find_approx_area(grad.get_timing(),grad_shape,(tf-post_delay,tf)),post_delay)])

        elif isinstance(grad,Gradient):
            post_delay = grad.get_duration() - (adc_dur + delay)
            if post_delay < 0 :
                raise ValueError("Delay + ADC sampling period is longer than gradient duration :(")
            self.fpwg([Gradient(delay*grad.get_amplitude(),delay)]) # delay before ADC
            signal = self.flat_readout(grad.get_amplitude(),adc) # flat readout
            self.fpwg([Gradient(post_delay*grad.get_amplitude(),post_delay)]) # delay after ADC

        return signal


# Helper methods
def anyrot(v):
    """ Returns Rodrigues's formula: 3 x 3 matrix for arbitrary rotation
        Around the direction of v and for norm(v) rads
        inputs
            v: 3-tuple (vx,vy,vz)
    """
    vx = v[0]
    vy = v[1]
    vz = v[2]
    th = np.linalg.norm(v,2)
    C = np.cos(th)
    S = np.sin(th)

    R = (1/(th*th))*np.array([[vx*vx*(1-C)+th*th*C, vx*vy*(1-C)-th*vz*S, vx*vz*(1-C)+th*vy*S],
                              [vx*vy*(1-C)+th*vz*S, vy*vy*(1-C)+th*th*C, vy*vz*(1-C)-th*vx*S],
                              [vx*vz*(1-C)-th*vy*S, vy*vz*(1-C)+th*vx*S, vz*vz*(1-C)+th*th*C]])
    return R


def find_approx_area(timing,level,interval):
    """
    Calculates area between interval[0] and interval[1] in the sequence "level" timed by "timing"
        using the trapezoid rule
    """
    t1 = interval[0]
    t2 = interval[1]
    a = np.where(timing<t1)[0][-1]
    b = np.where(timing>=t1)[0][0]
    c = np.where(timing<t2)[0][-1]
    d = np.where(timing>=t2)[0][0]

    new_timing = np.concatenate(([t1],timing[b:c+1],[t2]))
    x1 = level[a] + (level[b]-level[a])*(t1-timing[a])/(timing[b]-timing[a])
    x2 = level[c] + (level[d]-level[c])*(t2-timing[c])/(timing[d]-timing[c])
    new_level = np.concatenate(([x1],level[b:c+1],[x2]))

    return np.trapz(new_level,new_timing)


def get_scaled_gradient(grad,s):
    """
    Get an amplitude-scaled version of grad (without changing grad itself)
        INPUTS: grad - gradient to be scaled; s - scaling constant
    """
    if not isinstance(grad, Gradient):
        raise TypeError("Input 1 must be a gradient!")
    new_grad = copy.deepcopy(grad)
    new_grad.scale(s)
    return new_grad


