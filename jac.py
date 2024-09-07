import matplotlib.pyplot as plt
import numpy as np
import pyeit.eit.jac as jac
import pyeit.mesh as mesh
from pyeit.eit.fem import EITForward
from pyeit.eit.interp2d import sim2pts
from pyeit.mesh.shape import thorax
import pyeit.eit.protocol as protocol
from pyeit.mesh.wrapper import PyEITAnomaly_Circle
from scipy.interpolate import griddata

""" 0. build mesh """
n_el = 16  # number of electrodes
use_customize_shape = False
if use_customize_shape:
    mesh_obj = mesh.create(n_el, h0=0.1, fd=thorax)
else:
    mesh_obj = mesh.create(n_el, h0=0.1)

# Extract node, element, alpha
# Ensure pts is 2D
pts = mesh_obj.node[:, :2]  # Extract only the x and y coordinates
tri = mesh_obj.element
x, y = mesh_obj.node[:, 0], mesh_obj.node[:, 1]

# Remove duplicate points
pts, unique_indices = np.unique(pts, axis=0, return_index=True)
x = x[unique_indices]
y = y[unique_indices]

""" 1. problem setup """
anomaly = PyEITAnomaly_Circle(center=[0.5, 0.5], r=0.1, perm=1000.0)
mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly)

""" 2. FEM simulation """
protocol_obj = protocol.create(n_el, dist_exc=8, step_meas=1, parser_meas="std")

# Calculate simulated data
fwd = EITForward(mesh_obj, protocol_obj)
v0 = fwd.solve_eit()
v1 = fwd.solve_eit(perm=mesh_new.perm)

""" 3. JAC solver """
eit = jac.JAC(mesh_obj, protocol_obj)
eit.setup(p=0.5, lamb=0.01, method="kotre", perm=1, jac_normalized=True)
ds = eit.solve(v1, v0, normalize=True)
ds_n = sim2pts(pts, tri, np.real(ds))

# Remove duplicates from delta_perm to match pts length
delta_perm = mesh_new.perm - mesh_obj.perm
delta_perm = delta_perm[unique_indices]  # Filter delta_perm by unique_indices

# Create a regular grid to interpolate the data
grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]

# Interpolate the data onto the grid using 'linear' method
grid_z = griddata(pts, ds_n, (grid_x, grid_y), method='linear')
grid_delta_perm = griddata(pts, delta_perm, (grid_x, grid_y), method='linear')

# Plot ground truth
fig, axes = plt.subplots(1, 2, constrained_layout=True)
fig.set_size_inches(9, 4)

# Plot ground truth using imshow
ax = axes[0]
im1 = ax.imshow(grid_delta_perm.T, extent=(min(x), max(x), min(y), max(y)), origin='lower', vmin=np.min(delta_perm), vmax=np.max(delta_perm))
ax.set_aspect('equal')

# Plot EIT reconstruction using imshow
ax = axes[1]
im2 = ax.imshow(grid_z.T, extent=(min(x), max(x), min(y), max(y)), origin='lower', vmin=np.min(ds_n), vmax=np.max(ds_n))
for i, e in enumerate(mesh_obj.el_pos):
    ax.annotate(str(i + 1), xy=(x[e], y[e]), color="r")
ax.set_aspect("equal")

fig.colorbar(im2, ax=axes.ravel().tolist())
plt.show()
