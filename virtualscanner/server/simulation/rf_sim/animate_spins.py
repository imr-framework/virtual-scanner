# Animate spins using the SpinGroup class and stored magnetization evolution

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import FuncAnimation
from math import pi
from matplotlib import rc


###############################################################################
# Animation of multiple spins
def animate_spins(m_data, acc_factor=1, xyzlines=True, view_fig=True, save_fig=False, save_path='',title='Spin moves'):
    """

    Parameters
    ----------
    m_data : np.ndarray
        Magnetization in time of multiple spins to be animated
        Dim 1 : number of spins
        Dim 2 : time point
        Dim 3 : x/y/z

    acc_factor : int, optional
        Acceleration factor for frame-skipping.
        For example, a value of 3 retains every third frame
        Default is 1, which skips no frames.

    xyzlines : bool, optional
        Whether to include coordinate axes in the animated plot.
        Default is True.

    view_fig : bool, optional
        Whether to display the 3D animation
        Default is True.

    save_fig : bool, optional
        Whether to save the gif file
        Default is False.

    save_path : str, optional
        Where to save the gif file
        Default is current folder.

    title : str, optional
        Title of animated plot.
        Default is "Spin moves"

    """


    rc('animation', html='html5')

    # m_data: dim1=#spins, dim2=timepoint, dim3=Mx/My/Mz
    Nspins = np.shape(m_data)[0]
    Nframes = np.shape(m_data)[1]

    if np.shape(m_data)[2] != 3:
        raise ValueError("m_data should have 3 rows corresponding to Mx, My, and Mz")


    # Initialize figure and set properties
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.set_title(title)
    ax.set_xlim(-1, 1), ax.set_xticks([])
    ax.set_ylim(-1, 1), ax.set_yticks([])
    ax.set_zlim(-1, 1), ax.set_zticks([])


    if xyzlines:
        ax.plot([-1, 1], [0, 0], [0, 0], color='black', linestyle='dashed', linewidth=1)
        ax.plot([0, 0], [-1, 1], [0, 0], color='black', linestyle='dashed', linewidth=1)
        ax.plot([0, 0], [0, 0], [-1, 1], color='black', linestyle='dashed', linewidth=1)

    # Plot initial
    #line = ax.plot([0, m_data[0, 0]], [0, m_data[1, 0]], [0, m_data[2, 0]])[0]

    lines = [ax.plot([0,m_data[k,0,0]],[0,m_data[k,0,1]],[0,m_data[k,0,2]])[0] for k in range(Nspins)]
    # Define update function
    def update(k):
        """Helper for animate_spins"""
        # Retrieve correct frame
        for u in range(Nspins):
            M = m_data[u,(acc_factor*k+1)%Nframes,:]
            lines[u].set_data(np.array([[0,M[0]], [0, M[1]]]))  # set x and y data
            lines[u].set_3d_properties(np.array([0, M[2]]))  # set z data
        return lines

    # Make animation
    animation = FuncAnimation(fig, update, frames=round(Nframes/acc_factor), interval=1)

    # Save animation if directed
    if save_fig:
        print('Saving gif....')
        animation.save(save_path+'spin_moves.gif', writer='imagemagick', fps=60, dpi=80)

    # Display animation if directed
    if view_fig: plt.show()

    return animation 


# Example
if __name__ == '__main__':
    # Load some m data
    m_data1 = np.array([[0, 1, 0],
                        [0, 0, 1],
                        [1, 0, 0]])
    m_data2 = np.array([[1,0,0],[0,1,0],[0,0,1]])
    m_data = np.squeeze([m_data1, m_data2])
    print(np.shape(m_data))
    # Animate!
    animate_spins(m_data=np.squeeze([m_data1,m_data2]), xyzlines=True, view_fig=True)

