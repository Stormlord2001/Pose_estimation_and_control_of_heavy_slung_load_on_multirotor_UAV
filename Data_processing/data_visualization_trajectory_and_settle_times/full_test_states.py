import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Rosbag contains:
#time,xL_gt,yL_gt,zL_gt,vxL_gt,vyL_gt,vzL_gt,qvec_x_gt,qvec_y_gt,qvec_z_gt,qvec_wx_gt,qvec_wy_gt,qvec_wz_gt,xD_gt,yD_gt,zD_gt,xD_S,yD_S,zD_S,xL_S,yL_S,zL_S,qD_x,qD_y,qD_z,qD_w,qL_x,qL_y,qL_z,qL_w,thrust,body_rate_x,body_rate_y,body_rate_z,mode
#path = 'data/estop_nmpc_done/run5.csv'
#path = 'data/estop_px4_done/run2.csv'

path = 'data/full_system/run15.csv'

# Check if file path contains px4
px4_data = False
if 'px4' in path:
    print("Detected PX4 rosbag data format.")
    px4_data = True


# Only get data while in offboard mode
data = np.genfromtxt(path, delimiter=',', skip_header=1)
data = data[data[:, -1] == 1]

# Print the shape of the data
# print("Data shape:", data.shape)

# Make the x and y means line up with the setpoint means
data[:, 1] -= np.mean(data[:, 1]) - np.mean(data[:, 19])
#data[:, 2] += np.mean(data[:, 2]) - np.mean(data[:, 20])

# Only grab rows where the load setpoint is (0,0,0.75) 
# data = data[(data[:, 19] == 0.0) & (data[:, 20] == 0.0) & (data[:, 21] == 0.75)]

# Print the shape of the data
# print("Data shape:", data.shape)

states = np.zeros(data.shape[0])
current_state = 0
for i in range(1, data.shape[0]):
    if not px4_data:
        if not np.array_equal(data[i, 19:22], [0.0, 0.0, 0.75]):
           current_state = 0
        else:
            current_state = 1
    else:
        if not np.array_equal(data[i, 16:19], [0.0, 0.0, 3.75]):
            current_state = 0
        else:
            current_state = 1
    states[i] = current_state

state_changes = np.where(np.diff(states) != 0)[0] + 1
state_changes = np.insert(state_changes, 0, 0)  # Add the first index as a state change

# Remove data after the last state change
#if state_changes[-1] < data.shape[0]:
#    data = data[:state_changes[-1], :]  

# Remove data before the first state change
if state_changes[2] > 0:
    data = data[state_changes[2]:, :]
    states = states[state_changes[2]:]
    # Remove the first state change since we are starting from there
    state_changes = state_changes[2:] - state_changes[2]

data[:,0] -= data[0, 0]  # Normalize time to start at 0
time = data[:,0] # Ensure time vector matches data length
load_pose = data[:,1:4]
load_vel = data[:,4:7]
cable_vec = data[:,7:10]
w_cable_vec = data[:,10:13]
drone_pose = data[:,13:16]
drone_setpoint = data[:,16:19]
load_setpoint = data[:,19:22]
drone_attitude = data[:,22:26]
load_attitude = data[:,26:30]
thrust = data[:,30]
body_rate = data[:,31:34]
mode = data[:, -1]

if px4_data:
    # Subtract 3m in z-direction from drone setpoint position and set as load setpoint position
    load_setpoint[:, 0] = drone_setpoint[:, 0]
    load_setpoint[:, 1] = drone_setpoint[:, 1]
    load_setpoint[:, 2] = drone_setpoint[:, 2] - 3

if px4_data:
    load_pose[:, 0] -= np.mean(load_pose[:, 0]) - np.mean(load_setpoint[:, 0])
    load_pose[:, 1] -= np.mean(load_pose[:, 1]) - np.mean(load_setpoint[:, 1])
    load_pose[:, 2] -= np.mean(load_pose[:, 2]) - np.mean(load_setpoint[:, 2])

# Calculate settling time for x and y positions after each setpoint change
settling_times_x = []
settling_times_y = []
settling_index_x = []
settling_index_y = []
threshold = 0.2
time_threshold = 4.0
for i in range(len(state_changes)):
    # Check if the setpoint is (0.0,0.0,0.75)
    if not px4_data:
        if not np.array_equal(load_setpoint[state_changes[i], :], np.array([0.0, 0.0, 0.75])):
            continue
    else:
        if not np.array_equal(load_setpoint[state_changes[i], :], np.array([0.0, 0.0, 3.75])):
            continue

    start = state_changes[i]
    if i == len(state_changes) - 1:
        end = data.shape[0]
    else:
        end = state_changes[i + 1]

    time_within_threshold_x = 0
    time_within_threshold_y = 0
    settling_time_x = None
    settling_time_y = None
    first_index_x = None
    first_index_y = None
    for j in range(start, end):
        if abs(load_pose[j, 0] - load_setpoint[j, 0]) < threshold:
            time_within_threshold_x += time[j] - time[j-1]
            if first_index_x is None:
                first_index_x = j
        else:
            time_within_threshold_x = 0
            first_index_x = None
        if abs(load_pose[j, 1] - load_setpoint[j, 1]) < threshold:
            time_within_threshold_y += time[j] - time[j-1]
            if first_index_y is None:
                first_index_y = j
        else:
            time_within_threshold_y = 0
            first_index_y = None
        if time_within_threshold_x >= time_threshold and settling_time_x is None:
            settling_time_x = time[j] - time[start] - time_threshold
            settling_index_x.append(first_index_x)
        if time_within_threshold_y >= time_threshold and settling_time_y is None:
            settling_time_y = time[j] - time[start] - time_threshold
            settling_index_y.append(first_index_y)
    if settling_time_x is not None:
        settling_times_x.append(settling_time_x)
    if settling_time_y is not None:
        settling_times_y.append(settling_time_y)

print("Settling times for X position:", settling_times_x)
print("Settling times for Y position:", settling_times_y)

# Calculate relaxed settling time for x and y positions after each setpoint change
settling_times_x_relaxed = []
settling_times_y_relaxed = []
settling_index_x_relaxed = []
settling_index_y_relaxed = []
threshold = 0.2
time_threshold = 2.0
for i in range(len(state_changes)):
    # Check if the setpoint is (0.0,0.0,0.75)
    if not px4_data:
        if not np.array_equal(load_setpoint[state_changes[i], :], np.array([0.0, 0.0, 0.75])):
            continue
    else:
        if not np.array_equal(load_setpoint[state_changes[i], :], np.array([0.0, 0.0, 3.75])):
            continue

    start = state_changes[i]
    if i == len(state_changes) - 1:
        end = data.shape[0]
    else:
        end = state_changes[i + 1]

    time_within_threshold_x = 0
    time_within_threshold_y = 0
    settling_time_x = None
    settling_time_y = None
    first_index_x = None
    first_index_y = None
    
    for j in range(start, end):
        if abs(load_pose[j, 0] - load_setpoint[j, 0]) < threshold:
            time_within_threshold_x += time[j] - time[j-1]
            if first_index_x is None:
                first_index_x = j
        else:
            time_within_threshold_x = 0
            first_index_x = None
        if abs(load_pose[j, 1] - load_setpoint[j, 1]) < threshold:
            time_within_threshold_y += time[j] - time[j-1]
            if first_index_y is None:
                first_index_y = j
        else:
            time_within_threshold_y = 0
            first_index_y = None
        if time_within_threshold_x >= time_threshold and settling_time_x is None:
            settling_time_x = time[j] - time[start] - time_threshold
            settling_index_x_relaxed.append(first_index_x)
        if time_within_threshold_y >= time_threshold and settling_time_y is None:
            settling_time_y = time[j] - time[start] - time_threshold
            settling_index_y_relaxed.append(first_index_y)
    if settling_time_x is not None:
        settling_times_x_relaxed.append(settling_time_x)
    if settling_time_y is not None:
        settling_times_y_relaxed.append(settling_time_y)

print("Settling times for X position:", settling_times_x_relaxed)
print("Settling times for Y position:", settling_times_y_relaxed)


# Toggle this to only show the Y axis in the first plot
plot_only_y_axis = True

def add_duration_arrow(ax, start_time, end_time, y_offset, label):
        x_axis_transform = ax.get_xaxis_transform()
        ax.annotate(
            '',
            xy=(end_time, y_offset),
            xytext=(start_time, y_offset),
            xycoords=x_axis_transform,
            textcoords=x_axis_transform,
            arrowprops=dict(arrowstyle='|-|', color='black', lw=1.8, shrinkA=0, shrinkB=0),
            annotation_clip=False,
        )
        ax.text(
            (start_time + end_time) / 2,
            y_offset + 0.1,
            label,
            transform=x_axis_transform,
            ha='center',
            va='top',
            fontsize=13,
            color='black',
            clip_on=False,
        )

if plot_only_y_axis:
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Load Position during E-stop Experiment', fontsize=16)
    plot_config = [
        (ax, 1, 'Y', 'tab:orange', settling_index_y, settling_index_y_relaxed),
    ]
else:
    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle('Load Position during E-stop Experiment', fontsize=16)
    plot_config = [
        (axs[0], 0, 'X', 'tab:blue', settling_index_x, settling_index_x_relaxed),
        (axs[1], 1, 'Y', 'tab:orange', settling_index_y, settling_index_y_relaxed),
    ]

for ax, axis_index, axis_label, line_color, settling_indices, relaxed_settling_indices in plot_config:
    ax.plot(time, load_setpoint[:, axis_index], label=f'Load {axis_label} Setpoint', color='black', alpha=0.75, linewidth=2)
    ax.vlines(time[relaxed_settling_indices], color='tab:blue', linestyle='--', label=f'Load {axis_label} Relaxed Settling Time', ymin=-4, ymax=4, alpha=0.75)
    ax.vlines(time[settling_indices], color='tab:red', linestyle='--', label=f'Load {axis_label} Settling Time', ymin=-4, ymax=4, alpha=0.75)
    ax.plot(time, load_pose[:, axis_index], label=f'Load {axis_label} Position', color=line_color, linewidth=2)

    green_shaded = False
    red_shaded = False

    for i in range(1, data.shape[0]):
        if i == 1:
            if states[i] == 0:
                try:
                    ax.axvspan(data[i, 0], data[i + np.where(states[i:] == 1)[0][0], 0], color='gray', alpha=0.25)
                except IndexError:
                    ax.axvspan(data[i, 0], data[-1, 0], color='gray', alpha=0.25)
            else:
                try:
                    ax.axvspan(data[i, 0], data[i + np.where(states[i:] == 0)[0][0], 0], color='lightgray', alpha=0.25)
                except IndexError:
                    ax.axvspan(data[i, 0], data[-1, 0], color='lightgray', alpha=0.25)
        if states[i] != states[i-1]:
            if states[i] == 0:
                try:
                    ax.axvspan(data[i, 0], data[i + np.where(states[i:] == 1)[0][0], 0], color='gray', alpha=0.25, label='PX4 Controller' if not green_shaded else None)
                    green_shaded = True
                except IndexError:
                    ax.axvspan(data[i, 0], data[-1, 0], color='gray', alpha=0.25)
            else:
                try:
                    ax.axvspan(data[i, 0], data[i + np.where(states[i:] == 0)[0][0], 0], color='lightgray', alpha=0.25, label='NMPC Controller' if not red_shaded else None)
                    red_shaded = True
                except IndexError:
                    ax.axvspan(data[i, 0], data[-1, 0], color='lightgray', alpha=0.25)

    ax.set_ylabel(f'{axis_label} Position (m)', fontsize=14)

    # Make legend into a 2x3 grid above the plot
    ax.legend(fontsize=14, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3)

    #ax.legend(fontsize=12, loc='lower right')
    ax.set_xlim(0, time[-1])
    ax.set_ylim(-4, 4)
    ax.grid()

    mode_switch_times = time[state_changes]
    duration_spans = []
    if len(mode_switch_times) > 3:
        # Show the 2nd and 3rd mode durations only.
        duration_spans = [
            (mode_switch_times[1], mode_switch_times[2]),
            (mode_switch_times[2], mode_switch_times[3]),
        ]

    for arrow_index, (start_time, end_time) in enumerate(duration_spans):
        y_offset = 1
        add_duration_arrow(ax, start_time, end_time, y_offset, f'{end_time - start_time:.0f}s')


if plot_only_y_axis:
    ax.set_xlabel('Time (s)', fontsize=14)
else:
    axs[1].set_xlabel('Time (s)', fontsize=14)
plt.tight_layout()
plt.savefig('figures/full_system_full_y.png', dpi=300)
plt.show()

index = 9
start_index = state_changes[index] if len(state_changes) > index else 0
end_index = state_changes[index + 1] if len(state_changes) > index + 1 else data.shape[0]

fig, axs = plt.subplots(3, 2, figsize=(15, 9), sharex=True)

# Calculate the max range for the position for any axis
x_range = np.max(load_pose[start_index:end_index,0]) - np.min(load_pose[start_index:end_index,0])
y_range = np.max(load_pose[start_index:end_index,1]) - np.min(load_pose[start_index:end_index,1])
z_range = np.max(load_pose[start_index:end_index,2]) - np.min(load_pose[start_index:end_index,2])
max_range = max(x_range, y_range, z_range)*1.1

# Calculate a center value for each axis
x_center = (np.max(load_pose[start_index:end_index,0]) + np.min(load_pose[start_index:end_index,0])) / 2
y_center = (np.max(load_pose[start_index:end_index,1]) + np.min(load_pose[start_index:end_index,1])) / 2
z_center = (np.max(load_pose[start_index:end_index,2]) + np.min(load_pose[start_index:end_index,2])) / 2

# Create plot limits
x_limits = (x_center - max_range/2, x_center + max_range/2)
y_limits = (y_center - max_range/2, y_center + max_range/2)
z_limits = (z_center - max_range/2, z_center + max_range/2)

axs[0,0].plot(time[start_index:end_index], load_pose[start_index:end_index,0], label='Load X Position', color='tab:blue')
axs[0,0].axhline(load_setpoint[start_index, 0], color='r', linestyle='--', label=f'Load Setpoint X={load_setpoint[start_index, 0]:.2f}m')
axs[0,0].set_ylabel('X Position (m)', fontsize=14)
axs[0,0].legend()
axs[0,0].grid()
axs[0,0].set_ylim(x_limits)  # Set the same y-limits for all position plots
axs[0,0].set_title('Load Position ', fontsize=16)
axs[1,0].plot(time[start_index:end_index], load_pose[start_index:end_index,1], label='Load Y Position', color='tab:orange')
axs[1,0].axhline(load_setpoint[start_index, 1], color='r', linestyle='--', label=f'Load Setpoint Y={load_setpoint[start_index, 1]:.2f}m')
axs[1,0].set_ylabel('Y Position (m)', fontsize=14)
axs[1,0].legend()
axs[1,0].grid()
axs[1,0].set_ylim(y_limits)  # Set the same y-limits for all position plots
axs[2,0].plot(time[start_index:end_index], load_pose[start_index:end_index,2], label='Load Z Position', color='tab:green')
axs[2,0].axhline(load_setpoint[start_index, 2], color='r', linestyle='--', label=f'Load Setpoint Z={load_setpoint[start_index, 2]:.2f}m')
axs[2,0].set_ylabel('Z Position (m)', fontsize=14)
axs[2,0].set_xlabel('Time (s)', fontsize=14)
axs[2,0].legend()
axs[2,0].grid()
axs[2,0].set_ylim(z_limits)  # Set the same y-limits for all position plots
axs[2,0].set_xlim(time[start_index], time[end_index])

# Plot load velocity in individual 2d plots, ignore drone velocity
axs[0,1].plot(time[start_index:end_index], load_vel[start_index:end_index,0], label='Load X Velocity')
axs[0,1].plot(time[start_index:end_index], load_vel[start_index:end_index,1], label='Load Y Velocity')
axs[0,1].plot(time[start_index:end_index], load_vel[start_index:end_index,2], label='Load Z Velocity')
axs[0,1].set_ylabel('Load Velocity (m/s)', fontsize=14)
axs[0,1].legend()
axs[0,1].grid()
axs[0,1].set_title('Load Velocity ', fontsize=16)

# Plot cable vector components
# Normalize cable vector for better visualization
cable_vec_norm = np.linalg.norm(cable_vec[start_index:end_index], axis=1, keepdims=True)
cable_vec_normalized = cable_vec[start_index:end_index] / (cable_vec_norm + 1e-6)  # Add small value to avoid division by zero
axs[1,1].plot(time[start_index:end_index], cable_vec_normalized[:,0], label='Cable X Component')
axs[1,1].plot(time[start_index:end_index], cable_vec_normalized[:,1], label='Cable Y Component')
axs[1,1].plot(time[start_index:end_index], cable_vec_normalized[:,2], label='Cable Z Component')
axs[1,1].set_ylabel('Normalized Vector Components', fontsize=14)
axs[1,1].legend()
axs[1,1].grid()
axs[1,1].set_title('Cable Vector Components ', fontsize=16)

# Plot angular velocity of the payload vector
axs[2,1].plot(time[start_index:end_index], w_cable_vec[start_index:end_index,0], label='Angular Vel X')
axs[2,1].plot(time[start_index:end_index], w_cable_vec[start_index:end_index,1], label='Angular Vel Y')
axs[2,1].plot(time[start_index:end_index], w_cable_vec[start_index:end_index,2], label='Angular Vel Z')
axs[2,1].set_ylabel('Angular Velocity (rad/s)', fontsize=14)
axs[2,1].set_xlabel('Time (s)', fontsize=14)
axs[2,1].legend()
axs[2,1].grid()
axs[2,1].set_title('Angular Velocity of Payload Vector ', fontsize=16)
plt.tight_layout()
plt.savefig('figures/full_system_states.png', dpi=300)
plt.show()

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

# Use the 8th setpoint change again
if not px4_data:
    snapshot_times = np.divide(np.array([-20, 60, 150, 240]), 50)*240
    ax.plot(load_pose[start_index-96:end_index,0], load_pose[start_index-96:end_index,1], load_pose[start_index-96:end_index,2], label='Payload Trajectory')
    ax.plot(drone_pose[start_index-96:end_index,0], drone_pose[start_index-96:end_index,1], drone_pose[start_index-96:end_index,2], label='Drone Trajectory')
else:
    snapshot_times = np.divide(np.array([-10, 60, 230]), 50)*240
    ax.plot(load_pose[start_index-48:end_index,0], load_pose[start_index-48:end_index,1], load_pose[start_index-48:end_index,2], label='Payload Trajectory')
    ax.plot(drone_pose[start_index-48:end_index,0], drone_pose[start_index-48:end_index,1], drone_pose[start_index-48:end_index,2], label='Drone Trajectory')
snapshot_times = snapshot_times.astype(int) + start_index

for i in snapshot_times:
    # Rotations
    drone_quat = drone_attitude[i]
    drone_quat_w_first = np.array([drone_quat[3], drone_quat[0], drone_quat[1], drone_quat[2]])
    R_drone = quat_to_rot(drone_quat_w_first)
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
    ax.view_init(elev=10, azim=-180)

ax.set_xlabel('X Position (m)', fontsize=14)
ax.set_ylabel('Y Position (m)', fontsize=14)
ax.set_zlabel('Z Position (m)', fontsize=14)
ax.legend(fontsize=14, loc='lower right')

ax.set_title('3D Trajectory of Drone and Payload during Emergency Stop', fontsize=16)

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

plt.savefig('figures/full_system_3d_trajectory_with_snapshots.png', dpi=300)
plt.show()



# Make a plot of the commanded thrust and angular torques
fig, axs = plt.subplots(1, 2, figsize=(15, 3), sharex=True)
axs[0].plot(time[start_index:end_index], thrust[start_index:end_index], label='Throttle Command')
axs[0].set_ylabel('Throttle (%)', fontsize=14)
axs[0].legend()
axs[0].grid()
axs[0].set_title('Throttle Inputs', fontsize=16)
axs[0].set_xlabel('Time (s)', fontsize=14)
axs[1].plot(time[start_index:end_index], body_rate[start_index:end_index,0], label='Bodyrate X Command')
axs[1].set_ylabel('Bodyrate (rad/s)', fontsize=14)
axs[1].plot(time[start_index:end_index], body_rate[start_index:end_index,1], label='Bodyrate Y Command')
axs[1].plot(time[start_index:end_index], body_rate[start_index:end_index,2], label='Bodyrate Z Command')
axs[1].set_xlabel('Time (s)', fontsize=14)
axs[1].set_title('Bodyrate Inputs', fontsize=16)
axs[1].legend()
axs[1].grid()
plt.tight_layout()
plt.savefig('figures/full_system_control_inputs.png', dpi=300)
plt.show()





