import pandas as pd
import numpy as np
import os
from scipy.signal import savgol_filter
# from scipy.interpolate import make_interp_spline
from interpax import Interpolator1D

class TravelData():
    def __init__(self, data_path="data/metropolis/output"):
        ag_res = pd.read_parquet(os.path.join(data_path, "agent_results.parquet"))
        self.n = len(ag_res)
        ag_res.sort_values("arrival_time", inplace=True)
        self.arr_time = np.array(ag_res.arrival_time/3600)
        self.ag_ttime = np.array(ag_res.total_travel_time/3600)
        self.arr_time_0 = np.array(ag_res.arrival_time[ag_res.selected_alt_id == 0]/3600)
        self.ag_ttime_0 = np.array(ag_res.total_travel_time[ag_res.selected_alt_id == 0]/60)
        self.arr_time_1 = np.array(ag_res.arrival_time[ag_res.selected_alt_id == 1]/3600)
        self.ag_ttime_1 = np.array(ag_res.total_travel_time[ag_res.selected_alt_id == 1]/60)

    def make_func(self):
        init_points = self.n//2
        fin_points = self.n//2
        
        ag_ttime_ext = np.r_[
            (self.ag_ttime[0],)*init_points,
            self.ag_ttime,
            (self.ag_ttime[-1],)*init_points
             ]
        
        arr_time_ext = np.r_[
            np.linspace(0, self.arr_time[0], init_points),
            self.arr_time,
            np.linspace(0, self.arr_time[-1], fin_points)
            ]

        # ag_ttime_int
        y = savgol_filter(self.ag_ttime, self.n // 100, 1)
        x_int = np.linspace(0, 24, 800)
        y_int = np.interp(x_int, self.arr_time, y)
        # return make_interp_spline(x_int, y_int, k=2)
        return Interpolator1D(x_int, y_int, 'cubic2')
