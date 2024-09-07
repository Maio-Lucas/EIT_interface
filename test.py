import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# Example node positions (coordinates)
# For a 32x32 grid, you would typically have a list of coordinates.
x = np.linspace(0, 1, 32)
y = np.linspace(0, 1, 32)
X, Y = np.meshgrid(x, y)

# Flatten the grid coordinates
x_flat = X.flatten()
y_flat = Y.flatten()

# Create a Delaunay triangulation of the grid (this is a placeholder, actual mesh could be different)
triangulation = tri.Triangulation(x_flat, y_flat)

z = np.zeros_like(x_flat)

# Plot using tripcolor
plt.figure(figsize=(6, 6))
plt.gca().set_aspect('equal')
tripcolor_plot = plt.tripcolor(triangulation, z, shading='flat', vmin=-0.75, vmax=0.75, cmap='viridis')
plt.colorbar(tripcolor_plot)
plt.title('EIT Data Visualization using Tripcolor')
plt.show()
