from matplotlib import pyplot as plt

from scipy.optimize import minimize, bisect


import jax.numpy as jnp
from jax import jit
from jaxopt import GradientDescent
from jaxopt import Bisection

from timeit import timeit
import time
from tqdm import tqdm

from generate_data import generate_arrival
from travel_times import asymm_gaussian_plateau
from retrieve_data import likelihood, total_log_lik, total_liks

import numpy as np
#%%
num=1000
betas, gammas, ts, t_as = generate_arrival(num, travel_time=asymm_gaussian_plateau())#, mu_gamma=1.3)
lik_fun = jit(lambda mus: -total_log_lik(asymm_gaussian_plateau(), t_as)(mus[0], mus[1], mus[2], mus[3], mus[4]))

#%%
start_time = time.time()
res = minimize(lik_fun, (9., 2., 10., .5, .5), method="Nelder-Mead")
print(res.x)
print(f"{time.time() - start_time} seconds")
#%%
start_time = time.time()
solver = GradientDescent(lik_fun, stepsize=1e-5, maxiter=2000)
init = (.9, 2., 10., .5, .5)
val, state = jit(solver.run)(init)
print("finished optimizing")
print(jnp.array(val))
print(f"{time.time() - start_time} seconds")
#%%
x = jnp.linspace(6, 13, 1000)
liks_x = total_liks(asymm_gaussian_plateau(), x)(.7, 1.2, 9.5, .1, 1.)
#%%
h = 120
plt.hist(t_as, 300)
plt.fill_between(x, liks_x*h, alpha=.3, color="red")
plt.show()
#%%
liks = total_liks(asymm_gaussian_plateau(), t_as)(.7, 1.2, 9.5, .1, 1.)

#%%
fig, axs = plt.subplots(1, 2)
dis = axs[0].scatter(np.random.normal(size=num), t_as, s=10, c=liks/max(liks))
axs[0].set_ylabel("t_a")
axs[1].set_ylabel("t_a")
axs[1].set_xlabel("t*")
ord = axs[1].scatter(ts, t_as, s=10, c=liks/max(liks))
fig.colorbar(ord, shrink=.8, ticks=[])
fig.show()
