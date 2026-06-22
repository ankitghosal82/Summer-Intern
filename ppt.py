import numpy as np
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. Generate the Ellipsoid Manifold
# ---------------------------------------------------------
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

# Ellipsoid radii (W matrix representation)
rx, ry, rz = 1.0, 1.0, 1.5 

x = rx * np.outer(np.cos(u), np.sin(v))
y = ry * np.outer(np.sin(u), np.sin(v))
z = rz * np.outer(np.ones(np.size(u)), np.cos(v))
boundary_z = 0.8

# Create a color map for the surface: Safe vs Toxic
# We use custom scale values to clearly separate the regions
surfacecolor = np.where(z > boundary_z, 1, 0)

manifold = go.Surface(
    x=x, y=y, z=z,
    surfacecolor=surfacecolor,
    colorscale=[[0, 'lightblue'], [1, 'lightcoral']],
    showscale=False,
    opacity=0.6,
    name="Configuration Space"
)

# ---------------------------------------------------------
# 3. Simulate Trajectories
# ---------------------------------------------------------
t = np.linspace(0, 1, 100)

# Path 1: Uncontrolled Trajectory (Goes Toxic)
# Starts near equator, naturally drifts to the top pole
x_uncontrolled = 1.0 * np.cos(t * np.pi/2)
y_uncontrolled = 1.0 * np.sin(t * np.pi/2)
z_uncontrolled = 1.5 * np.sin(t * np.pi/2)

# Path 2: CBF Controlled Trajectory (Stays Safe)
# Follows natural path until boundary, then slides along h(x) = 0
x_safe = np.copy(x_uncontrolled)
y_safe = np.copy(y_uncontrolled)
z_safe = np.copy(z_uncontrolled)

# Apply CBF constraint: Clip Z to boundary and adjust X/Y to stay on ellipsoid
for i in range(len(t)):
    if z_safe[i] > boundary_z:
        z_safe[i] = boundary_z
        # Recalculate x, y to ensure we stay perfectly on the ellipsoid surface
        scale = np.sqrt((1 - (z_safe[i]/rz)**2) / ((x_safe[i]/rx)**2 + (y_safe[i]/ry)**2))
        x_safe[i] *= scale
        y_safe[i] *= scale

# ---------------------------------------------------------
# 4. Create the Plotly Figure
# ---------------------------------------------------------
fig = go.Figure(data=[manifold])

# Add the uncontrolled (Dangerous) trajectory
fig.add_trace(go.Scatter3d(
    x=x_uncontrolled, y=y_uncontrolled, z=z_uncontrolled,
    mode='lines',
    line=dict(color='red', width=6, dash='dash'),
    name='Uncontrolled Path (Toxic)'
))

# Add the CBF controlled (Safe) trajectory
fig.add_trace(go.Scatter3d(
    x=x_safe, y=y_safe, z=z_safe,
    mode='lines+markers',
    line=dict(color='green', width=8),
    marker=dict(size=4, color='green'),
    name='CBF Controlled Path (Safe)'
))

# Add the boundary line for visual clarity
theta = np.linspace(0, 2*np.pi, 100)
r_boundary = np.sqrt(1 - (boundary_z/rz)**2)
fig.add_trace(go.Scatter3d(
    x=rx * r_boundary * np.cos(theta),
    y=ry * r_boundary * np.sin(theta),
    z=np.full_like(theta, boundary_z),
    mode='lines',
    line=dict(color='black', width=4),
    name='Safety Boundary h(x) = 0'
))

# Formatting
fig.update_layout(
    title="Transformer Token Trajectory on State Manifold",
    scene=dict(
        xaxis_title='Feature 1',
        yaxis_title='Feature 2',
        zaxis_title='Toxicity Feature',
        camera=dict(eye=dict(x=1.5, y=1.5, z=0.5))
    ),
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

# Save to an HTML file you can open in a browser
fig.write_html("cbf_manifold_visualization.html")
fig.show()