import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


with h5py.File('sim_data.mat', 'r') as f:
    data = np.array(f['data'])


data = np.squeeze(data)
if data.shape[0] == 20:
    data = data.T
print("Data shape:", data.shape)

time = data[:,0]  # Ensure time vector matches data length
load_pose = data[:,1:4]
load_vel = data[:,4:7]
cable_vec = data[:,7:10]
w_cable_vec = data[:,10:13]
Drone_attitude = data[:,13:17]
Drone_angular_vel = data[:,17:20]

drone_pose = load_pose - cable_vec * 3.0

def plot_drone_detailed(ax, position, R, arm_length=0.2, prop_radius=0.08):
    # Arm endpoints in body frame
    arms = np.array([
        [-arm_length, 0, 0],   # back
        [ arm_length, 0, 0],   # front
        [0, -arm_length, 0],   # left
        [0,  arm_length, 0]    # right
    ])

    # Center
    center = np.array([0, 0, 0])

    # Transform arms
    arms_world = (R @ arms.T).T + position
    center_world = position

    # --- Draw arms ---
    # Arm endpoints in body frame (X configuration)
    d = arm_length / np.sqrt(2)

    arms = np.array([
        [ d,  d, 0],   # front-right
        [ d, -d, 0],   # front-left
        [-d, -d, 0],   # back-left
        [-d,  d, 0]    # back-right
    ])

    # Transform arms
    arms_world = (R @ arms.T).T + position
    center_world = position

    # --- Draw arms ---
    # Diagonal arm 1
    ax.plot([arms_world[0,0], arms_world[2,0]],
            [arms_world[0,1], arms_world[2,1]],
            [arms_world[0,2], arms_world[2,2]],
            'r-', linewidth=2)

    # Diagonal arm 2
    ax.plot([arms_world[1,0], arms_world[3,0]],
            [arms_world[1,1], arms_world[3,1]],
            [arms_world[1,2], arms_world[3,2]],
            'r-', linewidth=2)

    # --- Highlight front direction ---
    front = 0.5 * (arms_world[0] + arms_world[1])

    ax.plot([center_world[0], front[0]],
            [center_world[1], front[1]],
            [center_world[2], front[2]],
            'b-', linewidth=3)

    # --- Draw propellers as circles ---
    theta = np.linspace(0, 2*np.pi, 20)

    for i in range(4):
        rotor_center = arms[i]

        # Circle in XY plane of body frame
        circle = np.array([
            prop_radius * np.cos(theta),
            prop_radius * np.sin(theta),
            np.zeros_like(theta)
        ]).T + rotor_center

        # Transform to world frame
        circle_world = (R @ circle.T).T + position

        ax.plot(circle_world[:,0],
                circle_world[:,1],
                circle_world[:,2],
                'k-', linewidth=1)

def plot_payload(ax, position, R, side_length=0.75):
    half = side_length / 2

    # مربع corners in body frame (closed loop)
    corners = np.array([
        [-half, -half, 0],
        [ half, -half, 0],
        [ half,  half, 0],
        [-half,  half, 0],
        [-half, -half, 0]  # close the loop
    ])

    # Transform to world frame
    corners_world = (R @ corners.T).T + position

    # Plot square frame
    ax.plot(corners_world[:, 0],
            corners_world[:, 1],
            corners_world[:, 2],
            'g-', linewidth=2, label='_nolegend_', alpha=0.5)

def payload_rotation_from_cable(drone_pos, payload_pos):
    z_axis = payload_pos - drone_pos
    z_axis = z_axis / np.linalg.norm(z_axis)

    # Choose a reference vector that is NOT parallel to z_axis
    ref = np.array([1, 0, 0])
    if abs(np.dot(ref, z_axis)) > 0.9:
        ref = np.array([0, 1, 0])

    # Create orthonormal basis
    x_axis = np.cross(ref, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)

    # Rotation matrix (columns = body axes)
    R = np.column_stack((x_axis, y_axis, z_axis))
    return R

def quat_to_rot(q):
    x, y, z, w = q

    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x**2 + z**2),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])
    return R

def get_payload_corners(position, R, side_length=0.76):
    half = side_length / 2

    corners = np.array([
        [-half, -half, 0],
        [ half, -half, 0],
        [ half,  half, 0],
        [-half,  half, 0]
    ])

    corners_world = (R @ corners.T).T + position
    return corners_world

def rotation_matrix(roll, pitch, yaw):
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    Ry = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

ax.plot(load_pose[:,0], load_pose[:,1], load_pose[:,2], label='Payload Trajectory')
ax.plot(drone_pose[:,0], drone_pose[:,1], drone_pose[:,2], label='Drone Trajectory')
# Plot a small 3d model of the drone at 2 second intervals

snapshot_times = [0, 50, 80, 120, 170, 350]

for i in snapshot_times:
    # Rotations
    R_drone = quat_to_rot(Drone_attitude[i])
    R_payload = payload_rotation_from_cable(drone_pose[i], load_pose[i])

    # Plot drone + payload
    plot_drone_detailed(ax, drone_pose[i], R_drone, arm_length=0.25, prop_radius=0.125)
    plot_payload(ax, load_pose[i], R_payload)

    # Get payload corners in world frame
    corners_world = get_payload_corners(load_pose[i], R_payload)

    # Plot wires to each corner of the payload
    for corner in corners_world:
        ax.plot([drone_pose[i][0], corner[0]],
                [drone_pose[i][1], corner[1]],
                [drone_pose[i][2], corner[2]],
                'k', linewidth=0.66, alpha=0.7)

    # Plot red dot at the center of the payload
    ax.scatter(load_pose[i][0],
           load_pose[i][1],
           load_pose[i][2],
           color='red',
           s=20,
           depthshade=False)
    
    # Plot the time above the drone for each snapshot
    ax.text(drone_pose[i][0], drone_pose[i][1], drone_pose[i][2] + 0.3, f'{time[i]:.1f}s', color='black', fontsize=12, ha='center')

    # Constant camera angle
    ax.view_init(elev=20, azim=-135)

ax.set_xlabel('X Position (m)', fontsize=12)
ax.set_ylabel('Y Position (m)', fontsize=12)
ax.set_zlabel('Z Position (m)', fontsize=12)
ax.legend()

ax.set_title('3D Trajectory of Drone and Payload with Snapshots', fontsize=14)

# Get all data limits
x_limits = ax.get_xlim3d()
y_limits = ax.get_ylim3d()
z_limits = ax.get_zlim3d()

# Compute ranges
x_range = abs(x_limits[1] - x_limits[0])
y_range = abs(y_limits[1] - y_limits[0])
z_range = abs(z_limits[1] - z_limits[0])

# Find max range
max_range = max(x_range, y_range, z_range)

# Compute midpoints
x_mid = np.mean(x_limits)
y_mid = np.mean(y_limits)
z_mid = np.mean(z_limits)

# Set equal limits
ax.set_xlim3d([x_mid - max_range/2, x_mid + max_range/2])
ax.set_ylim3d([y_mid - max_range/2, y_mid + max_range/2])
ax.set_zlim3d([z_mid - max_range/2, z_mid + max_range/2])

# Remove white space around the 3D plot
plt.tight_layout()

plt.savefig('figures/3d_trajectory_with_snapshots.png', dpi=300)
plt.show()
