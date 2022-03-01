# Let's brainstorm ways of combining gradient and RF!
from math import pi,ceil
import numpy as np
import virtualscanner.server.simulation.bloch.spingroup_ps as sg
import matplotlib.pyplot as plt
from matplotlib.transforms import offset_copy

GAMMA = 42.58e6 * 2 * pi

def simulate_rf_STAR(b1, grad, dt, fov, n_spins, pdt1t2, group_df=0.0):
    # Simulates continuous rf!
    N = min(len(b1),len(grad))

    # Create list of spins
    #df_array = np.linspace(0, bw_spins/2, n_spins)

    xlocs = np.linspace(-fov/2, fov/2, n_spins)
    #
    #spins = [sg.SpinGroup(loc=(xloc,0,0), pdt1t2=pdt1t2, df=group_df) for xloc in xlocs]


    spins = [sg.NumSolverSpinGroup(loc=(xloc,0,0),pdt1t2=pdt1t2,df=group_df) for xloc in xlocs]
    all_signals=[spin.apply_rf_store(b1, grad, dt)[0] for spin in spins]

    return all_signals


def make_b1_grad(b1_mag, grad_mag, n, pad=0,opt="simultaneous"):
    zero_grads = np.zeros((3, n))
    nonzero_grads = grad_mag*np.tile([[1],[0],[0]], (1,n))
    b1_shape = b1_mag*np.ones(n)
    pad_grads = np.zeros((3,pad))
    pad_b1 = np.zeros(pad)
    # pad : number of zeros (for both RF and G) to pad at the end
    if opt == "Simultaneous": # length = n + pad
        b1 = np.concatenate((b1_shape,0*b1_shape, pad_b1))
        grad = np.concatenate((nonzero_grads,zero_grads, pad_grads),axis=1)

    elif opt == "RF_only": # length = n + pad
        b1 = np.concatenate((b1_shape,0*b1_shape, pad_b1))
        grad = np.concatenate((zero_grads,zero_grads, pad_grads),axis=1)

    elif opt == "RF_followed_by_G": # this option will have length 2*n + pad
        b1 = np.concatenate((b1_shape, np.zeros(n), pad_b1))
        grad = np.concatenate((zero_grads, nonzero_grads, pad_grads),axis=1)

    else:
        raise ValueError("option is not available; use Simultaneous, RF_only, or RF_followed_by_G")

    return b1, grad


def matrix_sim():
    # Tissue parameters
    types = {"GM": [1, 0.95, 0.1], "WM": [1, 0.6, 0.08], "CSF": [1, 4.5, 2.2]}


    # Different options, RF and gradient
    opts = ["Simultaneous", "RF_only", "RF_followed_by_G"]

    num_tp = 2000
    num_spins = 5

    fr_deg = 30
    flip_rate = (fr_deg * pi / 180) * 1e3  # 5 degrees per ms
    grad_mag = 5e-3  # T/m
    b1_mag = flip_rate / GAMMA
    all_signals = np.zeros((2 * num_tp, num_spins, len(types), len(opts)), dtype=complex)
    FOV = 0.256
    DT = 1e-7

    all_b1s = []
    all_grads = []

    for ind1, p in enumerate(types.keys()):  # For each tissue type
        for q in range(len(opts)):
            b1, grad = make_b1_grad(b1_mag, grad_mag, n=num_tp, pad=0, opt=opts[q])
            this_signal = simulate_rf_STAR(b1, grad, dt=DT, fov=FOV,
                                           n_spins=num_spins, pdt1t2=types[p], group_df=0)
            this_signal = np.array(this_signal)
            all_signals[:, :, ind1, q] = np.swapaxes(this_signal, 0, 1)
            all_b1s.append(b1)
            all_grads.append(grad)

    np.save("all_signals.npy", all_signals)
    tmodel = np.arange(0, 2 * num_tp * DT, DT)

    # Plot scheme
    tmodel = DT * np.arange(2 * num_tp)

    plt.figure(1)
    for u in range(3):
        plt.subplot(2, 3, u + 1)
        plt.plot(tmodel, all_b1s[u])
        plt.title("RF")
        plt.subplot(2, 3, u + 4)
        plt.plot(tmodel, all_grads[u][0, :])
        plt.title("Gx")

    # Plot signals

    cols = ["RF + grad", "RF only", "RF then grad"]
    rows = ["GM","WM","CSF"]

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
    plt.setp(axes.flat, xlabel='Time (ms)', ylabel='|Mxy|')
    pad = 5

    for ax, col in zip(axes[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

    for ax, row in zip(axes[:, 0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')

    fig.tight_layout()
    fig.subplots_adjust(left=0.15, top=0.95)

    all_signals = np.load("all_signals.npy")

    n_fig = 1
    for a in range(3):  # For each type
        for b in range(3):  # For each tissue
            ind_signals = all_signals[:, :, a, b]
            signal = np.absolute(np.sum(ind_signals, axis=1)) / num_spins
            axes[a,b].plot(1e3*tmodel, signal, label="Net signal")

            for c in range(np.shape(ind_signals)[1]):
                axes[a,b].plot(1e3*tmodel, np.absolute(ind_signals[:, c]), '--', label="Isochromat #" + str(c + 1))

            n_fig += 1

    print("Simulated with : ")
    print("# positions: ", str(num_spins))
    print("flip rate: ", str(fr_deg), " deg/ms")
    print("Gradient : ", str(grad_mag), " T/m")
    print("FOV: 256 mm")
    print("BW of spins: ", str(GAMMA*0.256*grad_mag/(2*pi)), " Hz")

    plt.legend()
    plt.show()


def single_sim():
    tissue = [1, 0.95, 0.1]
    # STAR parameters

    # Different options, RF and gradient
    opts = ["Simultaneous", "RF_only", "RF_followed_by_G"]
    q = 2


    num_tp = 10000
    num_spins = 5
    flip_rate = (20 * pi / 180) * 1e3  # 5 degrees per ms
    grad_mag = 0.1e-3  # T/m
    b1_mag = flip_rate / GAMMA  # % Flip rate of what?
    FOV = 0.256
    DT = 1e-6

    all_b1s = []
    all_grads = []

    b1, grad = make_b1_grad(b1_mag, grad_mag, n=num_tp, pad=0, opt=opts[q])
    this_signal = simulate_rf_STAR(b1, grad, dt=DT, fov=FOV,
                                           n_spins=5, pdt1t2=tissue, group_df=0)

    tmodel = np.arange(0, 2 * num_tp * DT, DT)

    # Plot scheme
    tmodel = DT * np.arange(2 * num_tp)

    #
    plt.figure(1)
    plt.plot(tmodel, np.absolute(this_signal[4]),'-b')
    plt.show()


def display_sim_results():

    # Display b1 and gradient
    plt.figure(1)


    # Display signals


if __name__ == "__main__":

    matrix_sim()