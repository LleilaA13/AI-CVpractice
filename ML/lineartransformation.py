import numpy as np
import matplotlib.pyplot as plt

def plot_grid(Xs, Ys, axs = None):
    t = np.arange(Xs.size)
    plt.plot(0, 0, marker = '*', markersize = 7, color = 'b', linestyle = 'none')
    plt.scatter(Xs, Ys, c = t, cmap = 'jet', marker = 'o')
    plt.axis('scaled')


# nX and nY are the boundaries of the grid in the x and y direction, and res is the resolution of the grid
nX, nY, res = 10, 10, 21
# creating linearly spaced arrays 'X' and 'Y' using np.linspace() from -nX to nX with 'res' points
X = np.linspace(-nX, +nX, res)
Y = np.linspace(-nY, +nY, res)

Xs, Ys = np.meshgrid(X, Y)

def linear_map(A, Xs, Ys):
    src = np.stack((Xs, Ys), axis = 2)
    src_r = src.reshape(-1, src.shape[-1])
    dst = A @ src_r.T
    dst = (dst.T).reshape(src.shape)
    return dst[:,:, 0], dst[:,:, 1]

ang = np.pi/4
A = np.array([[np.cos(ang), -np.sin(ang)],
            [np.sin(ang), np.cos(ang)]])
Xd, Yd = linear_map(A, Xs, Ys)
fig, axs = plt.subplots(1, 2)
fig.suptitle('Rotation')
plot_grid(Xs, Ys, axs[0])
plot_grid(Xd, Yd,axs[1])

plt.show()