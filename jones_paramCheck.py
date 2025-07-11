import os.path as op
import tempfile
import matplotlib.pyplot as plt

import hnn_core
from hnn_core import simulate_dipole, jones_2009_model
from hnn_core.viz import plot_dipole

net = jones_2009_model()
net.plot_cells()
net.cell_types['L4_stellate'].plot_morphology()

weights_ampa_d1 = {'L2_basket': 0.006562, 'L2_pyramidal': .000007, 'L4_stellate': 0.142300}
weights_nmda_d1 = {'L2_basket': 0.019482, 'L2_pyramidal': 0.004317, 'L4_stellate': 0.080074}
synaptic_delays_d1 = {'L2_basket': 0.1, 'L2_pyramidal': 0.1, 'L4_stellate': 0.1}
net.add_evoked_drive(
    'evdist1', mu=63.53, sigma=3.85, numspikes=1, weights_ampa=weights_ampa_d1,
    weights_nmda=weights_nmda_d1, location='distal',
    synaptic_delays=synaptic_delays_d1, event_seed=274)

weights_ampa_p1 = {'L2_basket': 0.08831, 'L2_pyramidal': 0.01525, 'L5_basket': 0.19934, 'L4_stellate': 0.00865}
synaptic_delays_prox = {'L2_basket': 0.1, 'L2_pyramidal': 0.1, 'L5_basket': 1., 'L4_stellate': 1.}
net.add_evoked_drive(
    'evprox1', mu=26.61, sigma=2.47, numspikes=1, weights_ampa=weights_ampa_p1,
    weights_nmda=None, location='proximal',
    synaptic_delays=synaptic_delays_prox, event_seed=544)

weights_ampa_p2 = {'L2_basket': 0.000003, 'L2_pyramidal': 1.438840, 'L5_basket': 0.008958, 'L4_stellate': 0.684013}
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

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 6), constrained_layout=True)
plot_dipole(dpls, ax=axes[0], layer='agg', show=False)
net.cell_response.plot_spikes_hist(ax=axes[1], spike_types=['evprox', 'evdist'])

plot_dipole(dpls, average=False, layer=['L2', 'L5', 'agg'], show=False)

net_sync = jones_2009_model()
n_drive_cells = 1
cell_specific = False

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