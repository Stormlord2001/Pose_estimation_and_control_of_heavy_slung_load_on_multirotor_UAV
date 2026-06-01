import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
from matplotlib.patches import Patch


# Get files from data folder
path = 'data/p2p_nmpc_done'

# Load the data from all csv files in the folder
csv_files = glob.glob(os.path.join(path, '*.csv')) 

settling_times_x_nmpc = []
settling_times_y_nmpc = []
settling_times_xy_nmpc = []

settling_times_x_px4 = []
settling_times_y_px4 = []
settling_times_xy_px4 = []

nmpc_succesfull_x = 0
nmpc_succesfull_y = 0
nmpc_succesfull_xy = 0

px4_succesfull_x = 0
px4_succesfull_y = 0
px4_succesfull_xy = 0

nmpc_total = 0
px4_total = 0

threshold = 0.20
time_threshold = 4.0

for file in csv_files:
    data = np.genfromtxt(file, delimiter=',', skip_header=1)
    # Make the x and y means line up with the setpoint means
    data[:, 1] -= np.mean(data[:, 1]) - np.mean(data[:, 19])
    data[:, 2] -= np.mean(data[:, 2]) - np.mean(data[:, 20])

    states = np.zeros(data.shape[0])
    current_state = 0
    for i in range(1, data.shape[0]):
        if not np.array_equal(data[i, 19:22], data[i-1, 19:22]):
            if current_state == 0:
                current_state = 1
            else:            
                current_state = 0
        states[i] = current_state

    state_changes = np.where(np.diff(states) != 0)[0] + 1
    state_changes = np.insert(state_changes, 0, 0)  # Add the first index as a state change

    # Remove data after the last state change
    if state_changes[-1] < data.shape[0]:
        data = data[:state_changes[-1], :]  

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

    # Calculate settling time for x and y positions after each setpoint change
    for i in range(len(state_changes)):
        start = state_changes[i]
        if i == len(state_changes) - 1:
            end = data.shape[0]
        else:
            end = state_changes[i + 1]
        time_within_threshold_x = 0
        time_within_threshold_y = 0
        time_within_threshold_xy = 0
        settling_time_x = None
        settling_time_y = None
        settling_time_xy = None
        first_index_x = None
        first_index_y = None
        first_index_xy = None
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
            # Check euclidean distance for both x and y
            if np.linalg.norm(load_pose[j, :2] - load_setpoint[j, :2]) < threshold:
                time_within_threshold_xy += time[j] - time[j-1]
            if time_within_threshold_x >= time_threshold and settling_time_x is None:
                settling_time_x = time[j] - time[start] - time_threshold
            if time_within_threshold_y >= time_threshold and settling_time_y is None:
                settling_time_y = time[j] - time[start] - time_threshold
            if time_within_threshold_xy >= time_threshold and settling_time_xy is None:
                settling_time_xy = time[j] - time[start] - time_threshold
        if settling_time_x is not None:
            settling_times_x_nmpc.append(settling_time_x)
            nmpc_succesfull_x += 1
        if settling_time_y is not None:
            settling_times_y_nmpc.append(settling_time_y)
            nmpc_succesfull_y += 1
        if settling_time_xy is not None:
            settling_times_xy_nmpc.append(settling_time_xy)
            nmpc_succesfull_xy += 1
        nmpc_total += 1
# Get files from data folder
path = 'data/p2p_px4_done'

# Load the data from all csv files in the folder
csv_files = glob.glob(os.path.join(path, '*.csv')) 

for file in csv_files:
    data = np.genfromtxt(file, delimiter=',', skip_header=1)
    # Make the x and y means line up with the setpoint means
    data[:, 1] -= np.mean(data[:, 1]) - np.mean(data[:, 19])
    data[:, 2] -= np.mean(data[:, 2]) - np.mean(data[:, 20])

    states = np.zeros(data.shape[0])
    current_state = 0
    for i in range(1, data.shape[0]):
        if not np.array_equal(data[i, 16:19], data[i-1, 16:19]):
            if current_state == 0:
                current_state = 1
            else:            
                current_state = 0
        states[i] = current_state

    state_changes = np.where(np.diff(states) != 0)[0] + 1
    state_changes = np.insert(state_changes, 0, 0)  # Add the first index as a state change

    # Remove data after the last state change
    if state_changes[-1] < data.shape[0]:
        data = data[:state_changes[-1], :]  

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

    load_setpoint[:, 0] = drone_setpoint[:, 0]
    load_setpoint[:, 1] = drone_setpoint[:, 1]
    load_setpoint[:, 2] = drone_setpoint[:, 2] - 3

    load_pose[:, 0] -= np.mean(load_pose[:, 0]) - np.mean(load_setpoint[:, 0])
    load_pose[:, 1] -= np.mean(load_pose[:, 1]) - np.mean(load_setpoint[:, 1])
    load_pose[:, 2] -= np.mean(load_pose[:, 2]) - np.mean(load_setpoint[:, 2])

    # Calculate settling time for x and y positions after each setpoint change
    for i in range(len(state_changes)):
        start = state_changes[i]
        if i == len(state_changes) - 1:
            end = data.shape[0]
        else:
            end = state_changes[i + 1]
        time_within_threshold_x = 0
        time_within_threshold_y = 0
        time_within_threshold_xy = 0
        settling_time_x = None
        settling_time_y = None
        settling_time_xy = None
        first_index_x = None
        first_index_y = None
        first_index_xy = None
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
            # Check euclidean distance for both x and y
            if np.linalg.norm(load_pose[j, :2] - load_setpoint[j, :2]) < threshold:
                time_within_threshold_xy += time[j] - time[j-1]
            else:
                time_within_threshold_xy = 0
            if time_within_threshold_x >= time_threshold and settling_time_x is None:
                settling_time_x = time[j] - time[start] - time_threshold
            if time_within_threshold_y >= time_threshold and settling_time_y is None:
                settling_time_y = time[j] - time[start] - time_threshold
            if time_within_threshold_xy >= time_threshold and settling_time_xy is None:
                settling_time_xy = time[j] - time[start] - time_threshold
        if settling_time_x is not None:
            settling_times_x_px4.append(settling_time_x)
            px4_succesfull_x += 1
        if settling_time_y is not None:
            settling_times_y_px4.append(settling_time_y)
            px4_succesfull_y += 1
        if settling_time_xy is not None:
            settling_times_xy_px4.append(settling_time_xy)
            px4_succesfull_xy += 1
        px4_total += 1

mean_x_nmpc = np.mean(settling_times_x_nmpc)
mean_y_nmpc = np.mean(settling_times_y_nmpc)
mean_xy_nmpc = np.mean(settling_times_xy_nmpc)
std_x_nmpc = np.std(settling_times_x_nmpc)
std_y_nmpc = np.std(settling_times_y_nmpc)
std_xy_nmpc = np.std(settling_times_xy_nmpc)

if settling_times_x_px4:
    mean_x_px4 = np.mean(settling_times_x_px4)
    std_x_px4 = np.std(settling_times_x_px4)
else:
    mean_x_px4 = 0
    std_x_px4 = 0

if settling_times_y_px4:
    mean_y_px4 = np.mean(settling_times_y_px4)
    std_y_px4 = np.std(settling_times_y_px4)
else:
    mean_y_px4 = 0
    std_y_px4 = 0

if settling_times_xy_px4:
    mean_xy_px4 = np.mean(settling_times_xy_px4)
    std_xy_px4 = np.std(settling_times_xy_px4)
else:
    mean_xy_px4 = 0
    std_xy_px4 = 0

print(f'NMPC: Mean Settling Time X: {mean_x_nmpc:.2f} s and standard deviation: {std_x_nmpc:.2f} s, total: {nmpc_total} and succesfull: {nmpc_succesfull_x}')
print(f'NMPC: Mean Settling Time Y: {mean_y_nmpc:.2f} s and standard deviation: {std_y_nmpc:.2f} s, total: {nmpc_total} and succesfull: {nmpc_succesfull_y}')
print(f'NMPC: Mean Settling Time XY: {mean_xy_nmpc:.2f} s and standard deviation: {std_xy_nmpc:.2f} s, total: {nmpc_total} and succesfull: {nmpc_succesfull_xy}')

print(f'PX4: Mean Settling Time X: {mean_x_px4:.2f} s and standard deviation: {std_x_px4:.2f} s, total: {px4_total} and succesfull: {px4_succesfull_x}')
print(f'PX4: Mean Settling Time Y: {mean_y_px4:.2f} s and standard deviation: {std_y_px4:.2f} s, total: {px4_total} and succesfull: {px4_succesfull_y}')
print(f'PX4: Mean Settling Time XY: {mean_xy_px4:.2f} s and standard deviation: {std_xy_px4:.2f} s, total: {px4_total} and succesfull: {px4_succesfull_xy}')

# Make a bar plot with standard deviation error bars for x, y and xy for both NMPC and PX4 with nmpc and px4 on same values on the x axis
labels = ['X-axis', 'Y-axis', 'Euclidean Distance']
x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots(figsize=(15, 5))
rects1 = ax.bar(x - width/2, [mean_x_nmpc, mean_y_nmpc, mean_xy_nmpc], width, yerr=[std_x_nmpc, std_y_nmpc, std_xy_nmpc], label='NMPC', capsize=5)
rects2 = ax.bar(x + width/2, [mean_x_px4, mean_y_px4, mean_xy_px4], width, yerr=[std_x_px4, std_y_px4, std_xy_px4], label='PX4', capsize=5)
# Plot the total and succesfull counts above the bars
for rect, total, succesfull in zip(rects1, [nmpc_total]*3, [nmpc_succesfull_x, nmpc_succesfull_y, nmpc_succesfull_xy]):
    height = rect.get_height()
    if succesfull == 0:
        ax.text(
            rect.get_x() + rect.get_width() * 0.76,
            height+0.5,
            f'{succesfull}/{total}',
            ha='center',
            va='bottom',
            fontsize=12,
            bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='black', lw=0.8, alpha=0.95)
        )
        ax.text(rect.get_x() + rect.get_width()*0.7, height, 'No settles detected', ha='center', va='bottom', fontsize=12)
    else:
       ax.text(
            rect.get_x() + rect.get_width() * 0.76,
            height,
            f'{succesfull}/{total}',
            ha='center',
            va='bottom',
            fontsize=12,
            bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='black', lw=0.8, alpha=0.95)
        )
for rect, total, succesfull in zip(rects2, [px4_total]*3, [px4_succesfull_x, px4_succesfull_y, px4_succesfull_xy]):
    height = rect.get_height()
    if succesfull == 0:
        ax.text(
            rect.get_x() + rect.get_width() * 0.76,
            height+0.5,
            f'{succesfull}/{total}',
            ha='center',
            va='bottom',
            fontsize=12,
            bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='black', lw=0.8, alpha=0.95)
        )
        ax.text(rect.get_x() + rect.get_width()*0.7, height, 'No settles detected', ha='center', va='bottom', fontsize=12)
    else:
       ax.text(
            rect.get_x() + rect.get_width() * 0.76,
            height,
            f'{succesfull}/{total}',
            ha='center',
            va='bottom',
            fontsize=12,
            bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='black', lw=0.8, alpha=0.95)
        )


ax.set_ylabel('Settling Time (s)',fontsize=14)
ax.xaxis.set_tick_params(labelsize=14)
ax.set_title('Mean Settling Times with Standard Deviation',fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(labels)

handles, labels_legend = ax.get_legend_handles_labels()
explain_patch = Patch(facecolor='white', edgecolor='black', label='xx settled out of yy attempts', alpha=0.95)
handles.append(explain_patch)
ax.legend(handles=handles, loc='upper right', fontsize=14)

plt.tight_layout()
#plt.savefig('figures/p2p_settling_times_comparison.png', dpi=300)
plt.show()