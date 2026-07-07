import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

# ==========================================
# 1. SYSTEM PARAMETERS
# ==========================================
start_pos = np.array([0.0, 0.0])
goal_pos = np.array([10.0, 10.0])

# Obstacle definition (The "Unsafe Set")
obs_pos = np.array([5.0, 5.0])
obs_radius = 2.0
safe_margin = 0.5 # Buffer zone
total_radius = obs_radius + safe_margin

# Control parameters
kp = 1.0       # Proportional gain for goal seeking
gamma = 2.0    # CBF strictness (\alpha or \gamma)
dt = 0.05      # Time step
steps = 400

# ==========================================
# 2. CBF CONTROLLER (The Math)
# ==========================================
def nominal_controller(p):
    """Open-loop goal seeking (like the base LLM prompt)."""
    return -kp * (p - goal_pos)

def solve_cbf_qp(p, u_nom):
    """
    Solves the constrained optimization using KKT conditions.
    No external solver required.
    """
    # h(x) = ||p - p_obs||^2 - R^2
    vector_to_obs = p - obs_pos
    distance_sq = np.dot(vector_to_obs, vector_to_obs)
    h_x = distance_sq - (total_radius)**2
    
    # \nabla h(x)
    grad_h = 2 * vector_to_obs
    
    # CBF Constraint: \nabla h * u >= -\gamma * h(x)
    # Formulated as: A * u <= b
    A = -grad_h
    b = gamma * h_x
    
    # Check if nominal control already satisfies safety
    if np.dot(A, u_nom) <= b:
        return u_nom, h_x
    
    # If unsafe, project u_nom onto the safe half-space boundary
    u_safe = u_nom - ((np.dot(A, u_nom) - b) / np.dot(A, A)) * A
    return u_safe, h_x

# ==========================================
# 3. RUN SIMULATION
# ==========================================
trajectory_open = []
trajectory_closed = []
h_x_history = []

p_open = start_pos.copy()
p_closed = start_pos.copy()

print("Simulating Open-Loop and Closed-Loop Trajectories...")
for _ in range(steps):
    # --- Open-Loop (Crashes into obstacle) ---
    u_open = nominal_controller(p_open)
    p_open = p_open + u_open * dt
    trajectory_open.append(p_open.copy())
    
    # --- Closed-Loop (Avoids obstacle via CBF) ---
    u_nom = nominal_controller(p_closed)
    u_safe, current_hx = solve_cbf_qp(p_closed, u_nom)
    p_closed = p_closed + u_safe * dt
    
    trajectory_closed.append(p_closed.copy())
    h_x_history.append(current_hx)
    
    # Stop if reached goal
    if np.linalg.norm(p_closed - goal_pos) < 0.1:
        break

traj_open = np.array(trajectory_open)
traj_closed = np.array(trajectory_closed)
h_x_hist = np.array(h_x_history)

# ==========================================
# 4. GENERATE PRESENTATION PLOTS
# ==========================================
print("Generating Plots...")
fig = plt.figure(figsize=(14, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1])

# --- Plot 1: The 2D Spatial Trajectory ---
ax1 = plt.subplot(gs[0])
ax1.set_title("Autonomous Navigation: CBF Obstacle Avoidance", fontsize=14, pad=15)

# Draw Obstacle
obstacle = patches.Circle(obs_pos, obs_radius, color='red', alpha=0.3, label='Unsafe Set (Obstacle)')
safe_boundary = patches.Circle(obs_pos, total_radius, color='orange', fill=False, linestyle='--', linewidth=2, label='Safety Boundary $h(x)=0$')
ax1.add_patch(obstacle)
ax1.add_patch(safe_boundary)

# Plot Trajectories
ax1.plot(traj_open[:, 0], traj_open[:, 1], 'r--', linewidth=2, label='Uncontrolled (Crash)')
ax1.plot(traj_closed[:, 0], traj_closed[:, 1], 'b-', linewidth=3, label='CBF Controlled (Safe)')

# Start and Goal points
ax1.plot(start_pos[0], start_pos[1], 'go', markersize=10, label='Start')
ax1.plot(goal_pos[0], goal_pos[1], 'g*', markersize=15, label='Goal')

ax1.set_xlim(-1, 11)
ax1.set_ylim(-1, 11)
ax1.set_xlabel("X Position", fontsize=12)
ax1.set_ylabel("Y Position", fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left')

# --- Plot 2: The Barrier Function h(x) over time ---
ax2 = plt.subplot(gs[1])
ax2.set_title(r"Control Barrier Function $h(x)$ Dynamics", fontsize=14, pad=15)

time_steps = np.arange(len(h_x_hist)) * dt
ax2.plot(time_steps, h_x_hist, 'b-', linewidth=3, label=r'System State $h(x)$')
ax2.axhline(0, color='black', linewidth=2, linestyle='-', label='Failure Boundary ($h(x)=0$)')

# Color safe and unsafe regions
ax2.fill_between(time_steps, 0, max(h_x_hist)+5, color='green', alpha=0.1, label='Safe Set $\mathcal{C}$')
ax2.fill_between(time_steps, min(h_x_hist)-5, 0, color='red', alpha=0.1, label='Unsafe Region')

ax2.set_xlabel("Time (s)", fontsize=12)
ax2.set_ylabel("Barrier Value $h(x)$", fontsize=12)
ax2.set_ylim(min(h_x_hist)-2, max(h_x_hist)+2)
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('cbf_physical_analogy.png', dpi=300)
print("Saved high-res presentation graphic as 'cbf_physical_analogy.png'.")
plt.show()