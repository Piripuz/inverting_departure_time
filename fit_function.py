import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from jax import grad, vmap
from jax import numpy as jnp
from scipy.optimize import curve_fit

import jax
jax.config.update('jax_enable_x64', True)


def left_hyp(a):
    return lambda x: (jnp.sqrt((x - a[2])**2 + a[0]) + x - a[2])/a[1]

def right_hyp(a):
    return lambda x: (jnp.sqrt((x - a[2])**2 + a[0]) - x + a[2])/a[1]

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


    coeff = np.r_[grad(left_hyp(a))(p[0]),
                  grad(right_hyp(c))(p[1]),
                  left_hyp(a)(p[0]),
                  b,
                  right_hyp(c)(p[1])]

    return np.linalg.solve(mat, coeff)

def func(a, b, c, p):
    bs = poly_coeffs(a, b, c, p)
    inner_func = lambda x: \
        jnp.piecewise(x,
                      [x < p[0], x > p[1]],
                      [
                          left_hyp(a),
                          right_hyp(c),
                          # lambda x: (x[:, None]**jnp.arange(len(bs))*bs).sum(axis=1)
                         lambda x: jnp.polyval(jnp.flip(bs), x)
                      ])
    return inner_func

def fit_to_data(x, y):
    to_fit = lambda x, a1, a2, a3, b1, c1, c2, c3, p1, p2: \
        func([a1, a2, a3], [b1], [c1, c2, c3], [p1, p2])(x)+2/60

    a_init = [.01, 2.5, 8.5]
    b_init = [y.max()]
    c_init = [.001, .7, 9.9]
    p_init = [9.3, 9.7]

    popt, _ = curve_fit(to_fit,
                     x,
                     y,
                     a_init + b_init + c_init + p_init,
                     bounds=(
                         [0, 0, -np.inf] + [-np.inf]*len(b_init) + [0, 0, -np.inf, 0, 0],
                         [np.inf]*(8+len(b_init))
                     ))
    a, b, c, p = popt[:3], popt[3:(3 + len(b_init))], popt[-5:-2], popt[-2:]
    return func(a, b, c, p)
