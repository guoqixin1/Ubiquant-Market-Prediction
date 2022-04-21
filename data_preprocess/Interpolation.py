import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as pt


def Interpolation(data_list):
    n = 100  # crank up
    i = np.arange(n, dtype=np.float64)

    # Chebyshev nodes:
    # nodes = np.cos((2*(i+1)-1)/(2*n)*np.pi)
    # Equispace nodes:
    nodes = np.linspace(-1, 1, n)

    V = np.cos(i*np.arccos(nodes.reshape(-1, 1)))
    coeffs = la.solve(V, data_list)

    x = np.linspace(-1, 1, 1000)
    Vfull = np.cos(i*np.arccos(x.reshape(-1, 1)))
    # pt.plot(x, np.dot(Vfull, coeffs))
    # pt.plot(nodes, data_list, "o")
    results = np.dot(Vfull, coeffs)
    return results
