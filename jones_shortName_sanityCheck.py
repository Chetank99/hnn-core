import os.path as op
import tempfile

import matplotlib.pyplot as plt
import hnn_core
from hnn_core import simulate_dipole, jones_2009_model
from hnn_core.viz import plot_dipole

net = jones_2009_model()
net.plot_cells()
# net.cell_types['L5Pyr'].plot_morphology().... no longer using this cell_types format
net.cell_types['L5Pyr']['object'].plot_morphology()

# Now you can use trying short_names in weights
weights_ampa_d1 = {'L2Basket': 0.006562, 'L2Pyr': .000007,
                   'L5Pyr': 0.142300}
weights_nmda_d1 = {'L2Basket': 0.019482, 'L2Pyr': 0.004317,
                   'L5Pyr': 0.080074}
synaptic_delays_d1 = {'L2Basket': 0.1, 'L2Pyr': 0.1,
                      'L5Pyr': 0.1}

net.add_evoked_drive(
    'evdist1', mu=63.53, sigma=3.85, numspikes=1, weights_ampa=weights_ampa_d1,
    weights_nmda=weights_nmda_d1, location='distal',
    synaptic_delays=synaptic_delays_d1, event_seed=274)

weights_ampa_p1 = {'L2Basket': 0.08831, 'L2Pyr': 0.01525,
                   'L5Basket': 0.19934, 'L5Pyr': 0.00865}
synaptic_delays_prox = {'L2Basket': 0.1, 'L2Pyr': 0.1,
                        'L5Basket': 1., 'L5Pyr': 1.}
# all NMDA weights are zero; pass None explicitly
net.add_evoked_drive(
    'evprox1', mu=26.61, sigma=2.47, numspikes=1, weights_ampa=weights_ampa_p1,
    weights_nmda=None, location='proximal',
    synaptic_delays=synaptic_delays_prox, event_seed=544)

# Second proximal evoked drive. NB: only AMPA weights differ from first
weights_ampa_p2 = {'L2Basket': 0.000003, 'L2Pyr': 1.438840,
                   'L5Basket': 0.008958, 'L5Pyr': 0.684013}
# all NMDA weights are zero; omit weights_nmda (defaults to None)
net.add_evoked_drive(
    'evprox2', mu=137.12, sigma=8.33, numspikes=1,
    weights_ampa=weights_ampa_p2, location='proximal',
    synaptic_delays=synaptic_delays_prox, event_seed=814)

from hnn_core import JoblibBackend

with JoblibBackend(n_jobs=2):
    dpls = simulate_dipole(net, tstop=170., n_trials=2)

window_len, scaling_factor = 30, 3000
for dpl in dpls:
    dpl.smooth(window_len).scale(scaling_factor)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 6),
                         constrained_layout=True)
plot_dipole(dpls, ax=axes[0], layer='agg', show=False)
net.cell_response.plot_spikes_hist(ax=axes[1],
                                   spike_types=['evprox', 'evdist'])

plot_dipole(dpls, average=False, layer=['L2', 'L5', 'agg'], show=False)

net_sync = jones_2009_model()

n_drive_cells=1
cell_specific=False

net_sync.add_evoked_drive(
    'evdist1', mu=63.53, sigma=3.85, numspikes=1, weights_ampa=weights_ampa_d1,
    weights_nmda=weights_nmda_d1, location='distal', n_drive_cells=n_drive_cells,
    cell_specific=cell_specific, synaptic_delays=synaptic_delays_d1, event_seed=274)

net_sync.add_evoked_drive(
    'evprox1', mu=26.61, sigma=2.47, numspikes=1, weights_ampa=weights_ampa_p1,
    weights_nmda=None, location='proximal', n_drive_cells=n_drive_cells,
    cell_specific=cell_specific, synaptic_delays=synaptic_delays_prox, event_seed=544)

net_sync.add_evoked_drive(
    'evprox2', mu=137.12, sigma=8.33, numspikes=1,
    weights_ampa=weights_ampa_p2, location='proximal', n_drive_cells=n_drive_cells,
    cell_specific=cell_specific, synaptic_delays=synaptic_delays_prox, event_seed=814)

print(net_sync.external_drives['evdist1']['dynamics'])

dpls_sync = simulate_dipole(net_sync, tstop=170., n_trials=1)

trial_idx = 0
dpls_sync[trial_idx].copy().smooth(window_len).scale(scaling_factor).plot()
net_sync.cell_response.plot_spikes_hist()


'''
import os.path as op
import hnn_core
from hnn_core import simulate_dipole, jones_2009_model
import matplotlib.pyplot as plt
from hnn_core.viz import plot_dipole, plot_psd

# Set up the network and initial proximal drive
net = jones_2009_model()

location = 'proximal'
burst_std = 20
weights_ampa_p = {'L2Pyr': 5.4e-5, 'L5Pyr': 5.4e-5}
syn_delays_p = {'L2Pyr': 0.1, 'L5Pyr': 1.}

net.add_bursty_drive(
    'alpha_prox', tstart=50., burst_rate=10, burst_std=burst_std, numspikes=2,
    spike_isi=10, n_drive_cells=10, location=location,
    weights_ampa=weights_ampa_p, synaptic_delays=syn_delays_p, event_seed=284)

# Simulate and plot the first figure (proximal drive only)
dpl = simulate_dipole(net, tstop=310., n_trials=1)
trial_idx = 0
dpl[trial_idx].scale(3000)

fig, axes = plt.subplots(2, 1, constrained_layout=True)
tmin, tmax = 10, 300

window_len = 20
dpl_smooth = dpl[trial_idx].copy().smooth(window_len)

dpl[trial_idx].plot(tmin=tmin, tmax=tmax, color='b', ax=axes[0], show=False)
dpl_smooth.plot(tmin=tmin, tmax=tmax, color='r', ax=axes[0], show=False)
axes[0].set_xlim((1, 399))

plot_psd(dpl[trial_idx], fmin=1., fmax=1e3, tmin=tmin, ax=axes[1], show=False)
axes[1].set_xscale('log')
plt.tight_layout()
plt.show() 

# Add the distal drive to the existing network
location = 'distal'
burst_std = 15
weights_ampa_d = {'L2Pyr': 5.4e-5, 'L5Pyr': 5.4e-5}
syn_delays_d = {'L2Pyr': 5., 'L5Pyr': 5.}
net.add_bursty_drive(
    'alpha_dist', tstart=50., burst_rate=10, burst_std=burst_std, numspikes=2,
    spike_isi=10, n_drive_cells=10, location=location,
    weights_ampa=weights_ampa_d, synaptic_delays=syn_delays_d, event_seed=296)

dpl = simulate_dipole(net, tstop=310., n_trials=1)

fig, axes = plt.subplots(3, 1, constrained_layout=True)

net.cell_response.plot_spikes_hist(ax=axes[0], show=False)

smooth_dpl = dpl[trial_idx].copy().smooth(window_len)

dpl[trial_idx].plot(tmin=tmin, tmax=tmax, ax=axes[1], color='b', show=False)
smooth_dpl.plot(tmin=tmin, tmax=tmax, ax=axes[1], color='r', show=False)

dpl[trial_idx].plot_psd(fmin=0., fmax=40., tmin=tmin, ax=axes[2], show=False)

plt.tight_layout()

plt.show()
'''
'''
import hnn_core
from hnn_core import jones_2009_model

net = jones_2009_model()

# filtering cell types based on metadata

# example 1....Get all pyramidal cells, regardless of layer
pyramidal_cells = net.filter_cell_types(morpho_type='pyramidal')
print(f"Pyramidal cells: {pyramidal_cells}")

# Example 2.....Get all cells in Layer 2
layer2_cells = net.filter_cell_types(layer='2')
print(f"Layer 2 cells: {layer2_cells}")

# Example 3.....Get only L5 basket cells (multiple criteria)...seeing if **kwargs works in this contextt
l5_basket_cells = net.filter_cell_types(morpho_type='basket', layer='5')
print(f"L5 basket cells: {l5_basket_cells}")

# Example 4.....edge caser testing...A filter that returns no matches
layer4_cells = net.filter_cell_types(layer='4')
print(f"Layer 4 cells: {layer4_cells}")
'''