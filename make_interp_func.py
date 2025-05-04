import numpy as np
import matplotlib.pyplot as plt

def left_hyp(a, der=0):
    if type(der) != int:
        raise Sese
    match der:
        case 0:
            return lambda x: (np.sqrt((x - a[2])**2 + a[0]) + x - a[2])/a[1]
        case 1:
            return lambda x: (x - a[2])/(a[1]*np.sqrt((x - a[2])**2 + a[0])) + 1/a[1]
        case _:
            raise NotImplementedError

def right_hyp(a, der=0):
    if type(der) != int:
        raise Sese
    match der:
        case 0:
            return lambda x: (np.sqrt((x - a[2])**2 + a[0]) - x + a[2])/a[1]
        case 1:
            return lambda x: (x - a[2])/(a[1]*np.sqrt((x - a[2])**2 + a[0])) - 1/a[1]
        case _:
            raise NotImplementedError        

def poly_coeffs(a, b, c, p):
    """Expect $a, c \in \mathbb{R}^3,
    b \in \mathbb{R}^{k-3},
    p \in \mathbb{R}^2$

    """

    k = len(b) + 3
    points = np.linspace(*p, k - 1)
    
    mat = np.eye(k+1)
    mat[0, :] = np.r_[0, np.arange(1, k+1) * p[0] ** np.arange(k)]
    mat[1, :] = np.r_[0, np.arange(1, k+1) * p[1] ** np.arange(k)]
    mat[2:, :] = points[:, None]**np.arange(k+1)


    coeff = np.r_[left_hyp(a, 1)(p[0]),
                  right_hyp(c, 1)(p[1]),
                  left_hyp(a)(p[0]),
                  b,
                  right_hyp(c)(p[1])]

    return np.linalg.solve(mat, coeff)

def func(a, b, c, p):
    bs = poly_coeffs(a, b, c, p)
    inner_func = lambda x: \
        np.piecewise(x,
                     [x < p[0], x > p[1]],
                     [
                         left_hyp(a),
                         right_hyp(c),
                         np.polynomial.polynomial.Polynomial(bs),
                     ])
    return inner_func


a = [.5, 20, 1]
b = [2, 2]
c = [.5, 2, 15]
p = [4, 10]

f = func(a, b, c, p)
x = np.linspace(-5, 18, 500)
plt.plot(x, f(x))
plt.show()
