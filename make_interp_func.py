import numpy as np
import matplotlib.pyplot as plt
from jax import numpy as jnp
from jax import grad
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


a = [.5, 20., 1.]
b = [2., 2.]
c = [.5, 2., 15.]
p = [4., 10.]

f = func(a, b, c, p)
x = np.linspace(p[0], p[1], 500)
# plt.plot(x, (x[:, None]**jnp.arange(len(bs))*bs).sum(axis=1))
plt.plot(x, f(x))
plt.show()
