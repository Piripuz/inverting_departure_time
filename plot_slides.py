import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Arc, Circle
import numpy as np
from generate_data import  generate_arrival, cost
from travel_times import asymm_gaussian_plateau, asymm_gaussian
from utils import TravelTime
from find_points import find_bs, find_gs, find_ts
from retrieve_data import likelihood, total_log_lik
from likelihood_split import total_liks_early, total_liks_kink, total_liks_late
from jax import vmap
import jax.numpy as jnp
from tqdm import tqdm
#%%

tt = TravelTime(asymm_gaussian_plateau(sigma_l=.7, sigma_r=.4, plateau_len=.4))
tt_cost = TravelTime(asymm_gaussian_plateau(sigma_l=.7, sigma_r=.4, plateau_len=.1))
tt_int = TravelTime(asymm_gaussian())

num = 1000
par = [.8, 1.4, 9.5, .3, 1]
np.random.seed(144)
betas, gammas, ts, t_as = generate_arrival(num, tt, *par)

bs = find_bs(par[0], tt)
gs = find_gs(par[1], tt)

bs_int = find_bs(par[0], tt_int)
gs_int = find_gs(par[1], tt_int)
ts_int = find_ts(par[0], par[1], tt_int)


early_color = "green"
late_color="red"
ot_color = "blue"
tt_color = "purple"
cost_color = "orange"
#%%
x = np.random.normal(size=num)
fig_scatter, ax_scatter = plt.subplots(figsize=(4, 5))
b_low = bs[0] - par[3]*1.7
b_high =bs[0] + par[3]*1.7
g_low = gs[1] - par[3]/2
g_high = gs[1] + par[3]/2
ax_scatter.fill_between([x.min(), x.max()], [b_low]*2, [b_high]*2, alpha=.1, color=early_color)
ax_scatter.fill_between([x.min(), x.max()], [g_low]*2, [g_high]*2, alpha=.1, color=late_color)
ax_scatter.scatter(x, t_as, s=2)
ax_scatter.set_xlim(x.min(), x.max())
ax_scatter.set_ylim(4, 15)
ax_scatter.set_xticks([])
ax_scatter.set_ylabel(r"$t_a$ (h)")
fig_scatter.savefig("slides/img/t_as.png", dpi=600)

y = np.linspace(4, 15, 500)
ax_scatter.plot(tt.f(y)*4 + x.min(), y, color="red")
fig_scatter.savefig("slides/img/t_as_tt.png", dpi=600)
plt.close(fig_scatter)

#%%

fig_bin, ax_bin = plt.subplots(figsize=(6, 4))
n, bins, patches = ax_bin.hist(t_as, 80)

ax_bin.set_yticks([])
ax_bin.set_xlabel(r"$t_a$ (h)")

def_color = patches[0].get_facecolor()

for b, p in zip(bins, patches):
    if b < b_high and b > b_low - p.get_width():
        p.set_facecolor(early_color)
    if b < g_high and b > g_low - p.get_width():
        p.set_facecolor(late_color)

x = np.linspace(6.5, 12.5, 300)
tt_line = ax_bin.plot(x, tt.f(x)*40, color=tt_color, linewidth=2, label="Travel time function")

labels = [tt_line[0].get_label(), "Early arrivals", "Late arrivals", "On time arrivals"]
handles = [tt_line[0], Patch(facecolor=early_color), Patch(facecolor=late_color), Patch(facecolor=def_color)]
ax_bin.legend(handles, labels)
ax_bin.set_xlim(6.5, 12.5)
fig_bin.savefig("slides/img/t_as_bins_tt.png", dpi=600)
plt.close(fig_bin)


#%%
def plot_zones(tt_internal):
    h_len = .4
    arc_len = .3
    text_dist = .15
    x = np.linspace(6.5, 12.5, 300)
    fig_tt, ax_tt = plt.subplots(figsize=(7, 4))
    ax_tt.plot(x, tt_internal.f(x), linewidth=2, color=tt_color, label='Travel time')

    bs = find_bs(par[0], tt_internal)
    gs = find_gs(par[1], tt_internal)
    ts = find_ts(par[0], par[1], tt_internal)

    ax_tt.set_xlabel(r"$t^*$ (h)")
    ax_tt.set_ylabel(r"$tt_a(t^*)$ (h)")

    tg_b = ax_tt.plot([bs[0], bs[1]], [tt_internal.f(bs[0]), tt_internal.f(bs[1])], color=early_color)
    h_b =  ax_tt.plot([bs[0], bs[0] + h_len], [tt_internal.f(bs[0])]*2, color=early_color)
    b_arc = Arc([bs[0], tt_internal.f(bs[0])], arc_len,
                arc_len*ax_tt.get_data_ratio()**(1/2), theta1=0,
                theta2=np.degrees(np.arctan(par[0])), color=early_color)
    ax_tt.text(bs[0] + text_dist, tt_internal.f(bs[0]) +
               text_dist*ax_tt.get_data_ratio()**(1/2),
               r"arctan$(\beta)$", size=8, color=early_color)
    ax_tt.add_patch(b_arc)

    tg_g = ax_tt.plot([gs[0], gs[1]], [tt_internal.f(gs[0]), tt_internal.f(gs[1])], color=late_color)
    h_g =  ax_tt.plot([gs[1] - h_len, gs[1]], [tt_internal.f(gs[1])]*2, color=late_color)
    g_arc = Arc([gs[1], tt_internal.f(gs[1])], arc_len,
                arc_len*ax_tt.get_data_ratio()**(1/2), angle=180, theta2=0,
                theta1=-np.degrees(np.arctan(par[1])), color=late_color)
    ax_tt.text(gs[1] - text_dist, tt_internal.f(gs[1]) +
               text_dist*ax_tt.get_data_ratio()**(1/2),
               r"arctan$(\gamma)$", size=8, color=late_color, ha='right')

    ax_tt.add_patch(g_arc)

    ax_tt.fill_between([bs[0], min([ts, bs[1]])], [ax_tt.get_ylim()[1]]*2, color=early_color, alpha=.2, label='Early arrival')
    ax_tt.fill_between([max([ts, gs[0]]), gs[1]], [ax_tt.get_ylim()[1]]*2, color=late_color, alpha=.2, label='Late arrival')

    ax_tt.legend()
    return fig_tt

plot_zones(tt).savefig("../seminar/slides/slides/img/tt_early_late.png", dpi=600)
plot_zones(tt_int).savefig("../seminar/slides/slides/img/tt_early_late_intersecting.png", dpi=600)
plt.close()

#%%

fig_mon, ax_mon = plt.subplots(figsize=(7, 4))

x = np.linspace(7.5, 11, 400)
def func(x):
    if x > bs_int[0] and x < ts_int:
        return bs_int[0]
    if x > ts_int and x < gs_int[1]:
        return gs_int[1]
    return x
y = np.vectorize(func)(x)
y[np.diff(y, prepend=y[0])>1] = np.nan
ax_mon.plot(x, y, c=tt_color, label = r"$t_a(t^*)$")
ymin = ax_mon.get_ylim()[0]
ymax = ax_mon.get_ylim()[1]
alpha = .1
ax_mon.fill_between([bs_int[0], ts_int], [ymin]*2, [ymax]*2, color=early_color, alpha=alpha, label='Early Arrivals')
ax_mon.fill_between([ts_int, gs_int[1]], [ymin]*2, [ymax]*2, color=late_color, alpha=alpha, label='Late Arrivals')
ax_mon.fill_between([gs_int[1], x[-1]], [ymin]*2, [ymax]*2, color=ot_color, alpha=alpha, label='On-time Arrivals')
ax_mon.fill_between([x[0], bs_int[0]], [ymin]*2, [ymax]*2, color=ot_color, alpha=alpha)

ax_mon.vlines([bs_int[0], gs_int[1]], ymin, [bs_int[0], gs_int[1]], [early_color, late_color], 'dashed')
ax_mon.text(bs_int[0] + .05, 7.6, r"$tt_a'(t^*) = \beta$", color=early_color)
ax_mon.text(gs_int[1] + .05, 8.3, r"$tt_a'(t^*) = -\gamma$", color=late_color)

ax_mon.legend(loc="upper left")
ax_mon.set_xlabel(r"$t^*$ (h)")
ax_mon.set_ylabel(r"$t_a$ (h)")
fig_mon.savefig("../seminar/slides/slides/img/monotone_t_a.png", dpi=600)
# fig_mon.show()
plt.close(fig_mon)
#%%
x = np.linspace(7, 11.5, 500)
y_early = total_liks_early(tt_int, x)(*par)
y_late = total_liks_late(tt_int, x)(*par)
y_kink = total_liks_kink(tt_int, x)(*par)

y_total = y_early + y_late + y_kink
#%%
fig_when, ax_when = plt.subplots(figsize=(7, 3))
fig_when.subplots_adjust(bottom=.15)
ax_when.fill_between(x, y_early, color=early_color, alpha=.2, label="Early Arrivals Likelihood")
ax_when.fill_between(x, y_late, color=late_color, alpha=.2, label="Late Arrivals Likelihood")
ax_when.fill_between(x, y_kink, color=ot_color, alpha=.2, label="On-Time Arrivals Likelihood")
ax_when.plot(x, tt_int.f(x)*2, color=tt_color, label=r"$tt_a(t_a)$")
ax_when.set_ylim(-0.2, 3)
ax_when.set_yticks([])
ax_when.legend()
ax_when.set_xlabel(r"$t_a$ (h)")
fig_when.savefig("../seminar/slides/slides/img/when_likelihood.png", dpi=600)

plt.close(fig_when)

#%%

fig_total, ax_total = plt.subplots(figsize=(7, 3))
ax_total.fill_between(x, y_total, alpha=.4, label="Total Likelihood")
ax_total.plot(x, tt_int.f(x)*2, color=tt_color, label=r"$tt_a(t_a)$")
ax_total.set_ylim(ax_total.get_ylim()[0], 4)
ax_total.set_yticks([])
ax_total.set_ylim(-0.2, 3)
ax_total.legend()
ax_total.set_xlabel(r"$t_a$ (h)")
fig_total.subplots_adjust(bottom=.15)
fig_total.savefig("../seminar/slides/slides/img/total_likelihood.png", dpi=600)
plt.close(fig_total)

#%%
_, _, _, points = generate_arrival(100000, tt, *par)
x = jnp.linspace(6, 13, 300)
ll = lambda x: likelihood(tt, x, *par)
lx = vmap(ll)(x)

#%%

fig_ll, ax_ll = plt.subplots(figsize=(6, 4))
ax_ll.hist(points, 400, label='Sampled points')
ax_ll.set_xlim(6, 12.5)
ax_ll.legend()
ax_ll.set_yticks([])
ax_ll.set_xlabel(r"$t_a$ (h)")
shade = ax_ll.fill_between(x, lx*3500, color="orange", alpha=.4, label=r'Likelihood of $t_a$')
shade.set_visible(False)
fig_ll.savefig("slides/img/hist_no_ll.png", dpi=600)
fig_ll.savefig("../seminar/slides/slides/img/hist_no_ll.png", dpi=600)
shade.set_visible(True)
ax_ll.legend()
fig_ll.savefig("slides/img/hist_ll.png", dpi=600)
fig_ll.savefig("../seminar/slides/slides/img/hist_ll.png", dpi=600)
plt.close(fig_ll)

#%%
num_ll = 1000
par = [.5, 1.3, 9.5, .3, 1]
_, _, _, t_as_ll = generate_arrival(num_ll, tt_int, *par)
ll = lambda x, y: total_log_lik(tt_int, t_as_ll)(x, y, *par[2:])
betas_contour =jnp.linspace(.01, .99, 200)
gammas_contour = jnp.linspace(1.01, 4, 200)
matrix_ll = vmap(vmap(ll, (0, None)), (None, 0))(betas_contour, gammas_contour) # vmap(ll_actual)(*m_contour)

#%%

fig_ct, ax_ct = plt.subplots(figsize=(6, 4))
X, Y = jnp.meshgrid(betas_contour, gammas_contour)
ct = ax_ct.contour(betas_contour, gammas_contour, matrix_ll, levels=50)
ax_ct.plot(par[0], par[1], 'or', label="Original Parameter Values")
ax_ct.set_xlabel(r"$\mu_\beta$")
ax_ct.set_ylabel(r"$\mu_\gamma$")
ax_ct.legend()
cbar = fig_ct.colorbar(ct)
cbar.set_label("Log Likelihood")
fig_ct.savefig("../seminar/slides/slides/img/contour_beautiful.png", dpi=600)
# fig_ct.savefig("slides/img/contour_beautiful.png", dpi=600)
fig_ct.show()
#%%
# par = (.4, 1.3, 9.5, .03, 1)
par[3]=.03
_, _, _, t_as_ll = generate_arrival(num_ll, tt, *par)

ll = lambda x, y: total_log_lik(tt, t_as_ll)(x, y, *par[2:])
betas_contour =jnp.linspace(.2, .99, 200)
gammas_contour = jnp.linspace(1.01, 4, 200)
matrix_ll = vmap(vmap(ll, (0, None)), (None, 0))(betas_contour, gammas_contour) # vmap(ll_actual)(*m_contour)

#%%

fig_ctb, ax_ctb = plt.subplots(figsize=(6, 4))
X, Y = jnp.meshgrid(betas_contour, gammas_contour)
ct = ax_ctb.contour(betas_contour, gammas_contour, matrix_ll, levels=50)
ax_ctb.plot(par[0], par[1], 'or', label="Original Parameter Values")
ax_ctb.legend()
ax_ctb.set_xlabel(r"$\mu_\beta$")
ax_ctb.set_ylabel(r"$\mu_\gamma$")
cbar = fig_ctb.colorbar(ct)
cbar.set_label("Log Likelihood")
fig_ctb.savefig("../seminar/slides/slides/img/contour_ugly.png", dpi=600)
fig_ctb.show()
# plt.close()

#%%
_, _, _, t_as_ll = generate_arrival(num_ll, tt, *par)

ll = lambda x, y: total_log_lik(tt, t_as_ll)(*par[:3], x, y)
sigmas_contour =jnp.linspace(.01, .99, 200)
sigmast_contour = jnp.linspace(.2, 4, 200)
matrix_ll_sigma = vmap(vmap(ll, (0, None)), (None, 0))(sigmas_contour, sigmast_contour) # vmap(ll_actual)(*m_contour)

#%%

fig_cts, ax_cts = plt.subplots(figsize=(6, 4))
X, Y = jnp.meshgrid(sigmas_contour, sigmast_contour)
ax_cts.contour(sigmas_contour, sigmast_contour, matrix_ll_sigma, levels=50)
ax_cts.plot(par[3], par[4], 'or')
ax_cts.set_xlabel(r"$\mu_\beta$")
ax_cts.set_ylabel(r"$\mu_\gamma$")
fig_cts.show()
#%%
num_ll = 1000
par = [.5, 1.3, 9.5, .6, 1]
_, _, _, t_as_ll = generate_arrival(num_ll, tt, *par)
ll = lambda x, y: total_log_lik(tt, t_as_ll)(x, y, *par[2:])
betas_contour =jnp.linspace(.01, .99, 200)
gammas_contour = jnp.linspace(1.01, 4, 200)
matrix_ll = vmap(vmap(ll, (0, None)), (None, 0))(betas_contour, gammas_contour) # vmap(ll_actual)(*m_contour)

#%%

fig_ct, ax_ct = plt.subplots(figsize=(6, 4))
X, Y = jnp.meshgrid(betas_contour, gammas_contour)
ct = ax_ct.contour(betas_contour, gammas_contour, matrix_ll, levels=50)
ax_ct.plot(par[0], par[1], 'or', label="Original Parameter Values")
ax_ct.set_xlabel(r"$\mu_\beta$")
ax_ct.set_ylabel(r"$\mu_\gamma$")
ax_ct.legend()
cbar = fig_ct.colorbar(ct)
cbar.set_label("Log Likelihood")
fig_ct.savefig("../seminar/slides/slides/img/contour_flat.png", dpi=600)
# fig_ct.savefig("slides/img/contour_beautiful.png", dpi=600)
fig_ct.show()

#%%

def plot_cost(b, g, star, cost_visible=True, draw_min=True):
    x_g = 10.5
    x_b = 1.8
    arc_len = .5
    text_dist = .4
    fig_cost, ax_cost = plt.subplots(figsize=(6, 4))
    x = np.linspace(6.5, 11.5, 300)
    cost_plot = ax_cost.plot(x, cost(tt_cost)(x, b, g, star), label=r"Cost function $C(t_a)$", color="orange")
    tt_plot = ax_cost.plot(x, tt_cost.f(x), label = r"Travel time $tt_a(t_a)$", color=tt_color)
    g_arc = Arc([x_g, cost(tt_cost)(x_g, b, g, star)], arc_len,
                arc_len*ax_cost.get_data_ratio()**(1/2), theta2=np.degrees(np.arctan(g)), color=late_color)
    b_arc = Arc([star - x_b, cost(tt_cost)(star - x_b, b, g, star)], arc_len,
                arc_len*ax_cost.get_data_ratio()**(1/2), angle=180, theta1=-np.degrees(np.arctan(b)), color=early_color)
    g_arc_patch = ax_cost.add_patch(g_arc)
    b_arc_patch = ax_cost.add_patch(b_arc)

    g_line = ax_cost.plot([x_g, x_g + arc_len], [cost(tt_cost)(x_g, b, g, star)]*2, color=late_color)
    b_line = ax_cost.plot([star - x_b, star - x_b - arc_len], [cost(tt_cost)(star - x_b, b, g, star)]*2, color=early_color)

    g_text = ax_cost.text(x_g + text_dist, cost(tt_cost)(x_g, b, g, star) +
                          text_dist*ax_cost.get_data_ratio(),
                          r"arctan$(\gamma)$", size=8, color=late_color, ha='left')

    b_text = ax_cost.text(star - x_b - text_dist, cost(tt_cost)(star - x_b, b, g, star) +
                          text_dist*ax_cost.get_data_ratio()**(2),
                          r"arctan$(\beta)$", size=8, color=early_color, ha='right')

    dashed_t = ax_cost.plot([star]*2, [tt_cost.f(star), 0], '--')
    t_text = ax_cost.text(star + .1, .1, r"$t^*$", size=10, ha="left", color=dashed_t[0].get_color())

    min_early = find_bs(b, tt_cost)[0]
    min_late = find_gs(g, tt_cost)[1]
    
    def argmin(d):
        if not d: return None
        min_val = min(d.values())
        return [k for k in d if d[k] == min_val][0]

    arr_time = argmin({
        min_early.item(): cost(tt_cost)(min_early, b, g, star),
        min_late.item(): cost(tt_cost)(min_late, b, g, star),
        star: tt_cost.f(star)})
    
    if arr_time == min_early:
        x_arr = min_early
        col = early_color
    elif arr_time ==  min_late:
        x_arr = min_late
        col = late_color
    elif arr_time == star:
        x_arr = star
        col = ot_color

    arr_dot = ax_cost.scatter(x_arr, cost(tt_cost)(x_arr, b, g, star), color=col, zorder=2.5)

    ax_cost.set_yticks([])
    ax_cost.set_xlabel(r"$t_a$ (h)")
    
    cost_plot[0].set_visible(cost_visible)
    g_arc.set_visible(cost_visible)
    b_arc.set_visible(cost_visible)
    g_arc_patch.set_visible(cost_visible)
    b_arc_patch.set_visible(cost_visible)
    g_line[0].set_visible(cost_visible)
    b_line[0].set_visible(cost_visible)
    g_text.set_visible(cost_visible)
    b_text.set_visible(cost_visible)
    dashed_t[0].set_visible(cost_visible)
    t_text.set_visible(cost_visible)
    arr_dot.set_visible(cost_visible*draw_min)

    if cost_visible:
        ax_cost.legend(loc="upper left")
    else:
        ax_cost.legend(handles=tt_plot, loc="upper left")

    ax_cost.set_ylim([None, 3])
    return fig_cost


# plot_cost(.6, 1.2, 9.4, False).savefig("../seminar/slides/slides/img/cost_only_tt.png", dpi=600)
plot_cost(.6, 1.2, 9.4, draw_min=False).savefig("../seminar/slides/slides/img/cost.png", dpi=600)
plot_cost(.6, 1.2, 9.4).savefig("../seminar/slides/slides/img/cost_early.png", dpi=600)
plot_cost(.6, 1.2, 9.67).savefig("../seminar/slides/slides/img/cost_late.png", dpi=600)
plot_cost(.95, 1.6, 9.5).savefig("../seminar/slides/slides/img/cost_ontime.png", dpi=600)
plt.close()

#%%
frames = 80
for i in tqdm(range(frames)):
    plot_cost(.95, 1.6, 9+(i/frames)*1.2).savefig(f"../seminar/slides/slides/img/animation/frame_{i}.png", dpi=600)
    plt.close()
plt.close()
#%%

fig_tt, ax_tt = plt.subplots(figsize=(6, 4))

x = np.linspace(6.5, 11.5, 300)
ax_tt.plot(x, tt_cost.f(x), label = r"Travel time $tt_a(t_a)$", color=tt_color)

ax_tt.set_xlabel(r"$t_a$ (h)")
ax_tt.set_ylabel(r"$tt_a(t_a)$ (h)")

fig_tt.savefig("../seminar/slides/slides/img/cost_only_tt.png", dpi=600)
fig_tt.show()
#%%

fig_ocost, ax_ocost = plt.subplots(figsize=(6, 3))

b, g, star = (.7, 1.3, 9.5)
x = np.linspace(8, 10.5, 300)
ax_ocost.plot(x, cost(tt_cost)(x, b, g, star), label=r"Cost function $C(t_a)$", color=cost_color)
ax_ocost.set_yticks([])
ax_ocost.set_xlabel(r"$t_a$ (h)")
ax_ocost.legend()

fig_ocost.subplots_adjust(bottom=.15)
fig_ocost.savefig("../seminar/slides/slides/img/cost_no_tt.png", dpi=600)

min_early = find_bs(b, tt_cost)[0]
min_late = find_gs(g, tt_cost)[1]
mins = np.r_[min_early, min_late, star]
ax_ocost.scatter(mins, cost(tt_cost)(mins, b, g, star), color=[early_color, late_color, ot_color], zorder=2.5)

fig_ocost.savefig("../seminar/slides/slides/img/cost_no_tt_mins.png", dpi=600)
plt.close()

#%%

n_0 = 10000
n_1 = 100

par_0 = (.8, 3.2)
par_1 = (.4, 1.4)

sigma_0 = .2
sigma_1 = .1

betas_0 = np.random.normal(par_0[0], sigma_0, n_0)
gammas_0 = np.random.normal(par_0[1], sigma_0, n_0)

betas_1 = np.random.normal(par_1[0], sigma_1, n_1)
gammas_1 = np.random.normal(par_1[1], sigma_1, n_1)

fig_distr, ax_distr = plt.subplots(figsize=(6, 4))

ax_distr.scatter(betas_0, gammas_0, s=.5, color="blue")
ax_distr.scatter(betas_1, gammas_1, s=.5, color="green")

circ_0 = Circle(par_0, sigma_0*3.5, ec="blue", fill=None)
circ_1 = Circle(par_1, sigma_1*3.5, ec="green", fill=None)
ax_distr.add_patch(circ_0)
ax_distr.add_patch(circ_1)

ax_distr.text(par_0[0] + sigma_0*3.5/np.sqrt(2) + .05, par_0[1] - sigma_0*3.5/np.sqrt(2) - .13, "Group 1", color="blue")
ax_distr.text(par_1[0] + sigma_1*3.5/np.sqrt(2) + .05, par_1[1] + sigma_1*3.5/np.sqrt(2) + .05, "Group 2", color="green")

ax_distr.set_xlabel(r"$\beta$")
ax_distr.set_ylabel(r"$\gamma$")

ax_distr.spines['top'].set_visible(False)
ax_distr.spines['right'].set_visible(False)

fig_distr.savefig("../seminar/slides/slides/img/distr_metr.png", dpi=600)
plt.close()
