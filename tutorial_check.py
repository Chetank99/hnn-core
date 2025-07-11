import matplotlib.pyplot as plt
from hnn_core import simulate_dipole
from hnn_core.viz import plot_dipole
from hnn_core.network_models import custom_cell_types_model

print("=== Testing with custom_cell_types_model ===")
net_custom = custom_cell_types_model(mesh_shape=(4, 4))
net_custom.plot_cells()

weights_ampa_d1_custom = {
    'L2_interneuron': 0.006562,
    'L2_pyramidal': .000007,
    'L4_stellate': 0.05,
    'L5_pyramidal': 0.142300
}
weights_nmda_d1_custom = {
    'L2_interneuron': 0.019482,
    'L2_pyramidal': 0.004317,
    'L4_stellate': 0.03,
    'L5_pyramidal': 0.080074
}
synaptic_delays_d1_custom = {
    'L2_interneuron': 0.1,
    'L2_pyramidal': 0.1,
    'L4_stellate': 0.5,
    'L5_pyramidal': 0.1
}

# Add distal drive
net_custom.add_evoked_drive(
    'evdist1', mu=63.53, sigma=3.85, numspikes=1,
    weights_ampa=weights_ampa_d1_custom,
    weights_nmda=weights_nmda_d1_custom,
    location='distal', synaptic_delays=synaptic_delays_d1_custom,
    event_seed=274)

# Define proximal weights
weights_ampa_p1_custom = {
    'L2_interneuron': 0.08831,
    'L2_pyramidal': 0.01525,
    'L4_stellate': 0.05,
    'L5_pyramidal': 0.00865
}
synaptic_delays_prox_custom = {
    'L2_interneuron': 0.1,
    'L2_pyramidal': 0.1,
    'L4_stellate': 0.5,
    'L5_pyramidal': 1.
}

# Add proximal drive
net_custom.add_evoked_drive(
    'evprox1', mu=26.61, sigma=2.47, numspikes=1,
    weights_ampa=weights_ampa_p1_custom,
    weights_nmda=None,
    location='proximal', synaptic_delays=synaptic_delays_prox_custom,
    event_seed=544)

# Simulate and plot
dpls_custom = simulate_dipole(net_custom, tstop=170., n_trials=1)
window_len, scaling_factor = 30, 3000
dpl = dpls_custom[0].copy().smooth(window_len).scale(scaling_factor)

# Plot dipole and spike histogram
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 6),
                         constrained_layout=True)
plot_dipole([dpl], ax=axes[0], layer='agg', show=False)
net_custom.cell_response.plot_spikes_hist(ax=axes[1],
                                          spike_types=['evprox', 'evdist'])

plt.suptitle('Custom Cell Types Model')
plt.show()
net_custom.cell_types['L4_stellate'].plot_morphology()