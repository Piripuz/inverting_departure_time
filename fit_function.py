import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from make_interp_func import func

ag_res = pd.read_parquet("data/metropolis/output/agent_results.parquet")
ag_res.sort_values("arrival_time", inplace=True)
arr_time = np.array(ag_res.arrival_time/3600)
ag_ttime = np.array(ag_res.total_travel_time/3600)

to_fit = lambda x, a1, a2, a3, b1, c1, c2, c3, p1, p2: \
    func([a1, a2, a3], [b1], [c1, c2, c3], [p1, p2])(x)+2/60

a_init = [.01, 2.5, 8.5]
b_init = [ag_ttime.max()]
c_init = [.001, .7, 9.9]
p_init = [9.3, 9.7]

popt, *others = curve_fit(to_fit,
                          arr_time,
                          ag_ttime,
                          a_init + b_init + c_init + p_init,
                          full_output=True,
                          bounds=(
                              [0, 0, -np.inf] + [-np.inf]*len(b_init) + [0, 0, -np.inf, 0, 0],
                              [np.inf]*(8+len(b_init))
                          ),
                          maxfev=10000)

a, b, c, p = popt[:3], popt[3:(3 + len(b_init))], popt[-5:-2], popt[-2:]
#%%
x = np.linspace(arr_time[0], arr_time[-1], 1000)
plt.scatter(arr_time, ag_ttime, s=1)
# plt.plot(x, func([.01, 2.5, 8.5], [0], [.001, .7, 9.9], [9.3, 9.7])(x)+2/60, color="red")
plt.vlines(p, 2/60, ag_ttime.max())

plt.plot(x, func(a, b, c, p)(x)+2/60, color="red")
plt.show()
