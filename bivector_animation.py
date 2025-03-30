import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Define basis vectors
e1 = np.array([1, 0, 0])
e2 = np.array([0, 1, 0])
origin = np.zeros(3)

# Rotation matrix for angle θ about the z-axis
def rotation_matrix(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])

# Define parallelogram points from two vectors
def parallelogram(v1, v2):
    return np.array([
        origin,
        v1,
        v1 + v2,
        v2,
        origin  # close the loop
    ])

# Set up figure and 3D axis
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Initialize plot elements
para_line, = ax.plot([], [], [], lw=2, color='red')
v1_arrow = ax.quiver(*origin, *e1, color='blue', arrow_length_ratio=0.1)
v2_arrow = ax.quiver(*origin, *e2, color='blue', arrow_length_ratio=0.1)

# Set axis limits and labels
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([0, 1])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Animation of Bivector e1 ∧ e2 Under Rotation')

# Initialization function
def init():
    para_line.set_data([], [])
    para_line.set_3d_properties([])
    return para_line,

# Animation function
def update(frame):
    theta = np.radians(frame)
    R = rotation_matrix(theta)

    v1_rot = R @ e1
    v2_rot = R @ e2
    para = parallelogram(v1_rot, v2_rot)

    para_line.set_data(para[:, 0], para[:, 1])
    para_line.set_3d_properties(para[:, 2])

    # Remove old arrows by removing all collections except the line
    for collection in ax.collections:
        if collection != para_line:
            collection.remove()
    
    # Draw new arrows
    ax.quiver(*origin, *v1_rot, color='blue', arrow_length_ratio=0.1)
    ax.quiver(*origin, *v2_rot, color='blue', arrow_length_ratio=0.1)

    return para_line,

# Create animation
ani = FuncAnimation(fig, update, frames=np.linspace(0, 90, 91), init_func=init,
                    blit=False, interval=50)

plt.tight_layout()
plt.show()
