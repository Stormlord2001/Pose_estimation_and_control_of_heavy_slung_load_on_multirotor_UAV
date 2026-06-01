import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
import os
import glob

# Get files from data folder
path = 'data/estop_nmpc_done'

# Load the data from all csv files in the folder
csv_files = glob.glob(os.path.join(path, '*.csv')) 

settling_times_y_nmpc = []
settling_times_xy_nmpc = []

settling_times_y_full = []
settling_times_xy_full = []

settling_times_y_px4 = []
settling_times_xy_px4 = []

nmpc_succesfull_y = 0
nmpc_succesfull_xy = 0

full_succesfull_y = 0
full_succesfull_xy = 0

px4_succesfull_y = 0
px4_succesfull_xy = 0

nmpc_total = 0
full_total = 0
px4_total = 0

threshold = 0.20
time_threshold = 4.0

#########################################

settling_times_y_nmpc_relaxed = []
settling_times_xy_nmpc_relaxed = []

settling_times_y_full_relaxed = []
settling_times_xy_full_relaxed = []

settling_times_y_px4_relaxed = []
settling_times_xy_px4_relaxed = []

nmpc_succesfull_y_relaxed = 0
nmpc_succesfull_xy_relaxed = 0

full_succesfull_y_relaxed = 0
full_succesfull_xy_relaxed = 0

px4_succesfull_y_relaxed = 0
px4_succesfull_xy_relaxed = 0

threshold_relaxed = 0.2
time_threshold_relaxed = 2.0

###########################################

settling_times_y_nmpc_very_relaxed = []
settling_times_xy_nmpc_very_relaxed = []

settling_times_y_full_very_relaxed = []
settling_times_xy_full_very_relaxed = []

settling_times_y_px4_very_relaxed = []
settling_times_xy_px4_very_relaxed = []

nmpc_succesfull_y_very_relaxed = 0
nmpc_succesfull_xy_very_relaxed = 0

full_succesfull_y_very_relaxed = 0
full_succesfull_xy_very_relaxed = 0

px4_succesfull_y_very_relaxed = 0
px4_succesfull_xy_very_relaxed = 0

threshold_very_relaxed = 0.5
time_threshold_very_relaxed = 2.0

for file in csv_files:
    data = np.genfromtxt(file, delimiter=',', skip_header=1)
    # Make the x and y means line up with the setpoint means
    data[:, 1] -= np.mean(data[:, 1]) - np.mean(data[:, 19])
    data[:, 2] += np.mean(data[:, 2]) - np.mean(data[:, 20])

    states = np.zeros(data.shape[0])
    current_state = 0
    for i in range(1, data.shape[0]):
        if not np.array_equal(data[i, 19:22], [0.0, 0.0, 0.75]):
           current_state = 0
        else:
            current_state = 1
        states[i] = current_state

    state_changes = np.where(np.diff(states) != 0)[0] + 1
    state_changes = np.insert(state_changes, 0, 0)  # Add the first index as a state change

    # Remove data after the last state change
    #if state_changes[-1] < data.shape[0]:
    #    data = data[:state_changes[-1], :]  

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
        if not np.array_equal(load_setpoint[state_changes[i], :], np.array([0.0, 0.0, 0.75])):
            continue
        start = state_changes[i]
        if i == len(state_changes) - 1:
            end = data.shape[0]
        else:
            end = state_changes[i + 1]

        time_within_threshold_y = 0
        time_within_threshold_xy = 0

        time_withing_treshold_y_relaxed = 0
        time_withing_treshold_xy_relaxed = 0

        time_withing_treshold_y_very_relaxed = 0
        time_withing_treshold_xy_very_relaxed = 0

        settling_time_y = None
        settling_time_xy = None

        settling_time_y_relaxed = None
        settling_time_xy_relaxed = None

        settling_time_y_very_relaxed = None
        settling_time_xy_very_relaxed = None

        first_index_y = None
        first_index_xy = None
        for j in range(start, end):
            if abs(load_pose[j, 1] - load_setpoint[j, 1]) < threshold:
                time_within_threshold_y += time[j] - time[j-1]
            else:
                time_within_threshold_y = 0

            if abs(load_pose[j, 1] - load_setpoint[j, 1]) < threshold_relaxed:
                time_withing_treshold_y_relaxed += time[j] - time[j-1]
            else:
                time_withing_treshold_y_relaxed = 0

            if abs(load_pose[j, 1] - load_setpoint[j, 1]) < threshold_very_relaxed:
                time_withing_treshold_y_very_relaxed += time[j] - time[j-1]
            else:
                time_withing_treshold_y_very_relaxed = 0

            # Check euclidean distance for both x and y
            if np.linalg.norm(load_pose[j, :2] - load_setpoint[j, :2]) < threshold:
                time_within_threshold_xy += time[j] - time[j-1]
            else:
                time_within_threshold_xy = 0

            if np.linalg.norm(load_pose[j, :2] - load_setpoint[j, :2]) < threshold_relaxed:
                time_withing_treshold_xy_relaxed += time[j] - time[j-1]
            else:
                time_withing_treshold_xy_relaxed = 0
            
            if np.linalg.norm(load_pose[j, :2] - load_setpoint[j, :2]) < threshold_very_relaxed:
                time_withing_treshold_xy_very_relaxed += time[j] - time[j-1]
            else:
                time_withing_treshold_xy_very_relaxed = 0

            if time_within_threshold_y >= time_threshold and settling_time_y is None:
                settling_time_y = time[j] - time[start] - time_threshold
            if time_within_threshold_xy >= time_threshold and settling_time_xy is None:
                settling_time_xy = time[j] - time[start] - time_threshold

            if time_withing_treshold_y_relaxed >= time_threshold_relaxed and settling_time_y_relaxed is None:
                settling_time_y_relaxed = time[j] - time[start] - time_threshold_relaxed
            if time_withing_treshold_xy_relaxed >= time_threshold_relaxed and settling_time_xy_relaxed is None:
                settling_time_xy_relaxed = time[j] - time[start] - time_threshold_relaxed

            if time_withing_treshold_y_very_relaxed >= time_threshold_very_relaxed and settling_time_y_very_relaxed is None:
                settling_time_y_very_relaxed = time[j] - time[start] - time_threshold_very_relaxed
            if time_withing_treshold_xy_very_relaxed >= time_threshold_very_relaxed and settling_time_xy_very_relaxed is None:
                settling_time_xy_very_relaxed = time[j] - time[start] - time_threshold_very_relaxed

        if settling_time_y is not None:
            settling_times_y_nmpc.append(settling_time_y)
            nmpc_succesfull_y += 1
        if settling_time_xy is not None:
            settling_times_xy_nmpc.append(settling_time_xy)
            nmpc_succesfull_xy += 1

        if settling_time_y_relaxed is not None:
            settling_times_y_nmpc_relaxed.append(settling_time_y_relaxed)
            nmpc_succesfull_y_relaxed += 1
        if settling_time_xy_relaxed is not None:
            settling_times_xy_nmpc_relaxed.append(settling_time_xy_relaxed)
            nmpc_succesfull_xy_relaxed += 1

        if settling_time_y_very_relaxed is not None:
            settling_times_y_nmpc_very_relaxed.append(settling_time_y_very_relaxed)
            nmpc_succesfull_y_very_relaxed += 1
        if settling_time_xy_very_relaxed is not None:
            settling_times_xy_nmpc_very_relaxed.append(settling_time_xy_very_relaxed)
            nmpc_succesfull_xy_very_relaxed += 1

        nmpc_total += 1

# Get files from data folder
path = 'data/full_system'

# Load the data from all csv files in the folder
csv_files = glob.glob(os.path.join(path, '*.csv')) 

for file in csv_files:
    data = np.genfromtxt(file, delimiter=',', skip_header=1)
    # Make the x and y means line up with the setpoint means
    data[:, 1] -= np.mean(data[:, 1]) - np.mean(data[:, 19])
    #data[:, 2] += np.mean(data[:, 2]) - np.mean(data[:, 20])

    states = np.zeros(data.shape[0])
    current_state = 0
    for i in range(1, data.shape[0]):
        if not np.array_equal(data[i, 19:22], [0.0, 0.0, 0.75]):
            current_state = 0
        else:
            current_state = 1
        states[i] = current_state

    state_changes = np.where(np.diff(states) != 0)[0] + 1
    state_changes = np.insert(state_changes, 0, 0)  # Add the first index as a state change

    if state_changes[2] > 0:
        data = data[state_changes[2]:, :]
        states = states[state_changes[2]:]
        # Remove the first state change since we are starting from there
        state_changes = state_changes[2:] - state_changes[2]

    # Remove data after the last state change
    #if state_changes[-1] < data.shape[0]:
    #    data = data[:state_changes[-1], :]  

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
        if not np.array_equal(load_setpoint[state_changes[i], :], np.array([0.0, 0.0, 0.75])):
            continue

        start = state_changes[i]
        if i == len(state_changes) - 1:
            end = data.shape[0]
        else:
            end = state_changes[i + 1]
        time_within_threshold_y = 0
        time_within_threshold_xy = 0

        time_withing_treshold_y_relaxed = 0
        time_withing_treshold_xy_relaxed = 0

        time_withing_treshold_y_very_relaxed = 0
        time_withing_treshold_xy_very_relaxed = 0

        settling_time_y = None
        settling_time_xy = None

        settling_time_y_relaxed = None
        settling_time_xy_relaxed = None

        settling_time_y_very_relaxed = None
        settling_time_xy_very_relaxed = None

        first_index_y = None
        first_index_xy = None

        for j in range(start, end):
            if abs(load_pose[j, 1] - load_setpoint[j, 1]) < threshold:
                time_within_threshold_y += time[j] - time[j-1]
            else:
                time_within_threshold_y = 0

            if abs(load_pose[j, 1] - load_setpoint[j, 1]) < threshold_relaxed:
                time_withing_treshold_y_relaxed += time[j] - time[j-1]
            else:
                time_withing_treshold_y_relaxed = 0

            if abs(load_pose[j, 1] - load_setpoint[j, 1]) < threshold_very_relaxed:
                time_withing_treshold_y_very_relaxed += time[j] - time[j-1]
            else:
                time_withing_treshold_y_very_relaxed = 0

            # Check euclidean distance for both x and y
            if np.linalg.norm(load_pose[j, :2] - load_setpoint[j, :2]) < threshold:
                time_within_threshold_xy += time[j] - time[j-1]
            else:
                time_within_threshold_xy = 0
            
            if np.linalg.norm(load_pose[j, :2] - load_setpoint[j, :2]) < threshold_relaxed:
                time_withing_treshold_xy_relaxed += time[j] - time[j-1]
            else:
                time_withing_treshold_xy_relaxed = 0

            if np.linalg.norm(load_pose[j, :2] - load_setpoint[j, :2]) < threshold_very_relaxed:
                time_withing_treshold_xy_very_relaxed += time[j] - time[j-1]
            else:
                time_withing_treshold_xy_very_relaxed = 0

            if time_within_threshold_y >= time_threshold and settling_time_y is None:
                settling_time_y = time[j] - time[start] - time_threshold
            if time_within_threshold_xy >= time_threshold and settling_time_xy is None:
                settling_time_xy = time[j] - time[start] - time_threshold

            if time_withing_treshold_y_relaxed >= time_threshold_relaxed and settling_time_y_relaxed is None:
                settling_time_y_relaxed = time[j] - time[start] - time_threshold_relaxed
            if time_withing_treshold_xy_relaxed >= time_threshold_relaxed and settling_time_xy_relaxed is None:
                settling_time_xy_relaxed = time[j] - time[start] - time_threshold_relaxed

            if time_withing_treshold_y_very_relaxed >= time_threshold_very_relaxed and settling_time_y_very_relaxed is None:
                settling_time_y_very_relaxed = time[j] - time[start] - time_threshold_very_relaxed
            if time_withing_treshold_xy_very_relaxed >= time_threshold_very_relaxed and settling_time_xy_very_relaxed is None:
                settling_time_xy_very_relaxed = time[j] - time[start] - time_threshold_very_relaxed

        if settling_time_y is not None:
            settling_times_y_full.append(settling_time_y)
            full_succesfull_y += 1

        if settling_time_xy is not None:
            settling_times_xy_full.append(settling_time_xy)
            full_succesfull_xy += 1

        if settling_time_y_relaxed is not None:
            settling_times_y_full_relaxed.append(settling_time_y_relaxed)
            full_succesfull_y_relaxed += 1

        if settling_time_xy_relaxed is not None:
            settling_times_xy_full_relaxed.append(settling_time_xy_relaxed)
            full_succesfull_xy_relaxed += 1

        if settling_time_y_very_relaxed is not None:
            settling_times_y_full_very_relaxed.append(settling_time_y_very_relaxed)
            full_succesfull_y_very_relaxed += 1
        
        if settling_time_xy_very_relaxed is not None:
            settling_times_xy_full_very_relaxed.append(settling_time_xy_very_relaxed)
            full_succesfull_xy_very_relaxed += 1

        full_total += 1

# Get files from data folder
path = 'data/estop_px4_done'

# Load the data from all csv files in the folder
csv_files = glob.glob(os.path.join(path, '*.csv')) 

for file in csv_files:
    data = np.genfromtxt(file, delimiter=',', skip_header=1)
    # Make the x and y means line up with the setpoint means
    data[:, 1] -= np.mean(data[:, 1]) - np.mean(data[:, 19])
    data[:, 2] += np.mean(data[:, 2]) - np.mean(data[:, 20])

    states = np.zeros(data.shape[0])
    current_state = 0
    for i in range(1, data.shape[0]):
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
        if not np.array_equal(drone_setpoint[state_changes[i], :], np.array([0.0, 0.0, 3.75])):
            continue

        start = state_changes[i]
        if i == len(state_changes) - 1:
            end = data.shape[0]
        else:
            end = state_changes[i + 1]
        time_within_threshold_y = 0
        time_within_threshold_xy = 0

        time_within_threshold_y_relaxed = 0
        time_within_threshold_xy_relaxed = 0

        time_within_threshold_y_very_relaxed = 0
        time_within_threshold_xy_very_relaxed = 0

        settling_time_x = None
        settling_time_y = None
        settling_time_xy = None

        settling_time_y_relaxed = None
        settling_time_xy_relaxed = None

        settling_time_y_very_relaxed = None
        settling_time_xy_very_relaxed = None

        first_index_x = None
        first_index_y = None
        first_index_xy = None
        for j in range(start, end):
            if abs(load_pose[j, 1] - load_setpoint[j, 1]) < threshold:
                time_within_threshold_y += time[j] - time[j-1]
                if first_index_y is None:
                    first_index_y = j
            else:
                time_within_threshold_y = 0
                first_index_y = None

            if abs(load_pose[j, 1] - load_setpoint[j, 1]) < threshold_relaxed:
                time_within_threshold_y_relaxed += time[j] - time[j-1]
            else:
                time_within_threshold_y_relaxed = 0

            if abs(load_pose[j, 1] - load_setpoint[j, 1]) < threshold_very_relaxed:
                time_within_threshold_y_very_relaxed += time[j] - time[j-1]
            else:
                time_within_threshold_y_very_relaxed = 0

            # Check euclidean distance for both x and y
            if np.linalg.norm(load_pose[j, :2] - load_setpoint[j, :2]) < threshold:
                time_within_threshold_xy += time[j] - time[j-1]
            else:
                time_within_threshold_xy = 0

            if np.linalg.norm(load_pose[j, :2] - load_setpoint[j, :2]) < threshold_relaxed:
                time_within_threshold_xy_relaxed += time[j] - time[j-1]
            else:
                time_within_threshold_xy_relaxed = 0

            if np.linalg.norm(load_pose[j, :2] - load_setpoint[j, :2]) < threshold_very_relaxed:
                time_within_threshold_xy_very_relaxed += time[j] - time[j-1]
            else:
                time_within_threshold_xy_very_relaxed = 0

            if time_within_threshold_y >= time_threshold and settling_time_y is None:
                settling_time_y = time[j] - time[start] - time_threshold
            if time_within_threshold_xy >= time_threshold and settling_time_xy is None:
                settling_time_xy = time[j] - time[start] - time_threshold

            if time_within_threshold_y_relaxed >= time_threshold_relaxed and settling_time_y_relaxed is None:
                settling_time_y_relaxed = time[j] - time[start] - time_threshold_relaxed
            if time_within_threshold_xy_relaxed >= time_threshold_relaxed and settling_time_xy_relaxed is None:
                settling_time_xy_relaxed = time[j] - time[start] - time_threshold_relaxed

            if time_within_threshold_y_very_relaxed >= time_threshold_very_relaxed and settling_time_y_very_relaxed is None:
                settling_time_y_very_relaxed = time[j] - time[start] - time_threshold_very_relaxed
            if time_within_threshold_xy_very_relaxed >= time_threshold_very_relaxed and settling_time_xy_very_relaxed is None:
                settling_time_xy_very_relaxed = time[j] - time[start] - time_threshold_very_relaxed

        if settling_time_y is not None:
            settling_times_y_px4.append(settling_time_y)
            px4_succesfull_y += 1
        if settling_time_xy is not None:
            settling_times_xy_px4.append(settling_time_xy)
            px4_succesfull_xy += 1

        if settling_time_y_relaxed is not None:
            settling_times_y_px4_relaxed.append(settling_time_y_relaxed)
            px4_succesfull_y_relaxed += 1
        if settling_time_xy_relaxed is not None:
            settling_times_xy_px4_relaxed.append(settling_time_xy_relaxed)
            px4_succesfull_xy_relaxed += 1

        if settling_time_y_very_relaxed is not None:
            settling_times_y_px4_very_relaxed.append(settling_time_y_very_relaxed)
            px4_succesfull_y_very_relaxed += 1
        if settling_time_xy_very_relaxed is not None:
            settling_times_xy_px4_very_relaxed.append(settling_time_xy_very_relaxed)
            px4_succesfull_xy_very_relaxed += 1

        px4_total += 1

mean_y_nmpc = np.mean(settling_times_y_nmpc)
mean_xy_nmpc = np.mean(settling_times_xy_nmpc)
std_y_nmpc = np.std(settling_times_y_nmpc)
std_xy_nmpc = np.std(settling_times_xy_nmpc)

mean_y_nmpc_relaxed = np.mean(settling_times_y_nmpc_relaxed)
mean_xy_nmpc_relaxed = np.mean(settling_times_xy_nmpc_relaxed)
std_y_nmpc_relaxed = np.std(settling_times_y_nmpc_relaxed)
std_xy_nmpc_relaxed = np.std(settling_times_xy_nmpc_relaxed)

mean_y_nmpc_very_relaxed = np.mean(settling_times_y_nmpc_very_relaxed)
mean_xy_nmpc_very_relaxed = np.mean(settling_times_xy_nmpc_very_relaxed)
std_y_nmpc_very_relaxed = np.std(settling_times_y_nmpc_very_relaxed)
std_xy_nmpc_very_relaxed = np.std(settling_times_xy_nmpc_very_relaxed)

if settling_times_y_full:
    mean_y_full = np.mean(settling_times_y_full)
    std_y_full = np.std(settling_times_y_full)
else:
    mean_y_full = 0
    std_y_full = 0

if settling_times_xy_full:
    mean_xy_full = np.mean(settling_times_xy_full)
    std_xy_full = np.std(settling_times_xy_full)
else:
    mean_xy_full = 0
    std_xy_full = 0

if settling_times_y_full_relaxed:
    mean_y_full_relaxed = np.mean(settling_times_y_full_relaxed)
    std_y_full_relaxed = np.std(settling_times_y_full_relaxed)
else:
    mean_y_full_relaxed = 0
    std_y_full_relaxed = 0

if settling_times_xy_full_relaxed:
    mean_xy_full_relaxed = np.mean(settling_times_xy_full_relaxed)
    std_xy_full_relaxed = np.std(settling_times_xy_full_relaxed)
else:
    mean_xy_full_relaxed = 0
    std_xy_full_relaxed = 0

if settling_times_y_full_very_relaxed:
    mean_y_full_very_relaxed = np.mean(settling_times_y_full_very_relaxed)
    std_y_full_very_relaxed = np.std(settling_times_y_full_very_relaxed)
else:
    mean_y_full_very_relaxed = 0
    std_y_full_very_relaxed = 0

if settling_times_xy_full_very_relaxed:
    mean_xy_full_very_relaxed = np.mean(settling_times_xy_full_very_relaxed)
    std_xy_full_very_relaxed = np.std(settling_times_xy_full_very_relaxed)
else:
    mean_xy_full_very_relaxed = 0
    std_xy_full_very_relaxed = 0

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

if settling_times_y_px4_relaxed:
    mean_y_px4_relaxed = np.mean(settling_times_y_px4_relaxed)
    std_y_px4_relaxed = np.std(settling_times_y_px4_relaxed)
else:
    mean_y_px4_relaxed = 0
    std_y_px4_relaxed = 0

if settling_times_xy_px4_relaxed:
    mean_xy_px4_relaxed = np.mean(settling_times_xy_px4_relaxed)
    std_xy_px4_relaxed = np.std(settling_times_xy_px4_relaxed)
else:
    mean_xy_px4_relaxed = 0
    std_xy_px4_relaxed = 0

if settling_times_y_px4_very_relaxed:
    mean_y_px4_very_relaxed = np.mean(settling_times_y_px4_very_relaxed)
    std_y_px4_very_relaxed = np.std(settling_times_y_px4_very_relaxed)
else:
    mean_y_px4_very_relaxed = 0
    std_y_px4_very_relaxed = 0

if settling_times_xy_px4_very_relaxed:
    mean_xy_px4_very_relaxed = np.mean(settling_times_xy_px4_very_relaxed)
    std_xy_px4_very_relaxed = np.std(settling_times_xy_px4_very_relaxed)
else:
    mean_xy_px4_very_relaxed = 0
    std_xy_px4_very_relaxed = 0

print(f'PX4: Mean Settling Time Y: {mean_y_px4:.2f} s and standard deviation: {std_y_px4:.2f} s, total: {px4_total} and succesfull: {px4_succesfull_y}')
print(f'PX4: Mean Settling Time XY: {mean_xy_px4:.2f} s and standard deviation: {std_xy_px4:.2f} s, total: {px4_total} and succesfull: {px4_succesfull_xy}')

print(f'PX4 (Relaxed): Mean Settling Time Y: {mean_y_px4_relaxed:.2f} s and standard deviation: {std_y_px4_relaxed:.2f} s, total: {px4_total} and succesfull: {px4_succesfull_y_relaxed}')
print(f'PX4 (Relaxed): Mean Settling Time XY: {mean_xy_px4_relaxed:.2f} s and standard deviation: {std_xy_px4_relaxed:.2f} s, total: {px4_total} and succesfull: {px4_succesfull_xy_relaxed}')

print(f'PX4 (Very Relaxed): Mean Settling Time Y: {mean_y_px4_very_relaxed:.2f} s and standard deviation: {std_y_px4_very_relaxed:.2f} s, total: {px4_total} and succesfull: {px4_succesfull_y_very_relaxed}')
print(f'PX4 (Very Relaxed): Mean Settling Time XY: {mean_xy_px4_very_relaxed:.2f} s and standard deviation: {std_xy_px4_very_relaxed:.2f} s, total: {px4_total} and succesfull: {px4_succesfull_xy_very_relaxed}')

print(f'NMPC: Mean Settling Time Y: {mean_y_nmpc:.2f} s and standard deviation: {std_y_nmpc:.2f} s, total: {nmpc_total} and succesfull: {nmpc_succesfull_y}')
print(f'NMPC: Mean Settling Time XY: {mean_xy_nmpc:.2f} s and standard deviation: {std_xy_nmpc:.2f} s, total: {nmpc_total} and succesfull: {nmpc_succesfull_xy}')

print(f'NMPC (Relaxed): Mean Settling Time Y: {mean_y_nmpc_relaxed:.2f} s and standard deviation: {std_y_nmpc_relaxed:.2f} s, total: {nmpc_total} and succesfull: {nmpc_succesfull_y_relaxed}')
print(f'NMPC (Relaxed): Mean Settling Time XY: {mean_xy_nmpc_relaxed:.2f} s and standard deviation: {std_xy_nmpc_relaxed:.2f} s, total: {nmpc_total} and succesfull: {nmpc_succesfull_xy_relaxed}')

print(f'NMPC (Very Relaxed): Mean Settling Time Y: {mean_y_nmpc_very_relaxed:.2f} s and standard deviation: {std_y_nmpc_very_relaxed:.2f} s, total: {nmpc_total} and succesfull: {nmpc_succesfull_y_very_relaxed}')
print(f'NMPC (Very Relaxed): Mean Settling Time XY: {mean_xy_nmpc_very_relaxed:.2f} s and standard deviation: {std_xy_nmpc_very_relaxed:.2f} s, total: {nmpc_total} and succesfull: {nmpc_succesfull_xy_very_relaxed}')

print(f'Full System: Mean Settling Time Y: {mean_y_full:.2f} s and standard deviation: {std_y_full:.2f} s, total: {full_total} and succesfull: {full_succesfull_y}')
print(f'Full System: Mean Settling Time XY: {mean_xy_full:.2f} s and standard deviation: {std_xy_full:.2f} s, total: {full_total} and succesfull: {full_succesfull_xy}')

print(f'Full System (Relaxed): Mean Settling Time Y: {mean_y_full_relaxed:.2f} s and standard deviation: {std_y_full_relaxed:.2f} s, total: {full_total} and succesfull: {full_succesfull_y_relaxed}')
print(f'Full System (Relaxed): Mean Settling Time XY: {mean_xy_full_relaxed:.2f} s and standard deviation: {std_xy_full_relaxed:.2f} s, total: {full_total} and succesfull: {full_succesfull_xy_relaxed}')

print(f'Full System (Very Relaxed): Mean Settling Time Y: {mean_y_full_very_relaxed:.2f} s and standard deviation: {std_y_full_very_relaxed:.2f} s, total: {full_total} and succesfull: {full_succesfull_y_very_relaxed}')
print(f'Full System (Very Relaxed): Mean Settling Time XY: {mean_xy_full_very_relaxed:.2f} s and standard deviation: {std_xy_full_very_relaxed:.2f} s, total: {full_total} and succesfull: {full_succesfull_xy_very_relaxed}')

labels = ['Criteria 1', 'Criteria 2', 'Criteria 3']
x = np.arange(len(labels))
width = 0.125

# Place six bars symmetrically around each tick so none overlap.
bar_offsets = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]) * width

fig, ax = plt.subplots(figsize=(14, 6))

series = [
    ('PX4 - Y-axis', [mean_y_px4_very_relaxed, mean_y_px4_relaxed, mean_y_px4], [std_y_px4_very_relaxed, std_y_px4_relaxed, std_y_px4], bar_offsets[2], 'tab:green', 1.0, None, [px4_succesfull_y_very_relaxed, px4_succesfull_y_relaxed, px4_succesfull_y], px4_total),
    ('NMPC mocap - Y-axis', [mean_y_nmpc_very_relaxed, mean_y_nmpc_relaxed, mean_y_nmpc], [std_y_nmpc_very_relaxed, std_y_nmpc_relaxed, std_y_nmpc], bar_offsets[0], 'tab:blue', 1.0, None, [nmpc_succesfull_y_very_relaxed, nmpc_succesfull_y_relaxed, nmpc_succesfull_y], nmpc_total),
    ('NMPC UKF - Y-axis', [mean_y_full_very_relaxed, mean_y_full_relaxed, mean_y_full], [std_y_full_very_relaxed, std_y_full_relaxed, std_y_full], bar_offsets[1], 'tab:orange', 1.0, None, [full_succesfull_y_very_relaxed, full_succesfull_y_relaxed, full_succesfull_y], full_total),
    ('PX4 - Euclidean', [mean_xy_px4_very_relaxed, mean_xy_px4_relaxed, mean_xy_px4], [std_xy_px4_very_relaxed, std_xy_px4_relaxed, std_xy_px4], bar_offsets[5], 'tab:green', 1.0, '//', [px4_succesfull_xy_very_relaxed, px4_succesfull_xy_relaxed, px4_succesfull_xy], px4_total),
    ('NMPC mocap - Euclidean', [mean_xy_nmpc_very_relaxed, mean_xy_nmpc_relaxed, mean_xy_nmpc], [std_xy_nmpc_very_relaxed, std_xy_nmpc_relaxed, std_xy_nmpc], bar_offsets[3], 'tab:blue', 1.0, '//', [nmpc_succesfull_xy_very_relaxed, nmpc_succesfull_xy_relaxed, nmpc_succesfull_xy], nmpc_total),
    ('NMPC UKF - Euclidean', [mean_xy_full_very_relaxed, mean_xy_full_relaxed, mean_xy_full], [std_xy_full_very_relaxed, std_xy_full_relaxed, std_xy_full], bar_offsets[4], 'tab:orange', 1.0, '//', [full_succesfull_xy_very_relaxed, full_succesfull_xy_relaxed, full_succesfull_xy], full_total),
]

for label, means, stds, offset, color, alpha, hatch, successes, total in series:
    rects = ax.bar(
        x + offset,
        means,
        width,
        yerr=stds,
        label=label,
        color=color,
        alpha=alpha,
        hatch=hatch,
        edgecolor='black',
        capsize=5,
    )
    for rect, success in zip(rects, successes):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() * 0.5,
            height+0.1,
            f'{success}/{total}',
            ha='center',
            va='bottom',
            fontsize=12,
            bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='black', lw=0.8, alpha=0.95)
        )

ax.set_ylabel('Mean Settling Time (s)', fontsize=16)
ax.xaxis.set_tick_params(labelsize=14)
ax.set_title('Mean Settling Times for Emergency Stop Experiments', fontsize=18)
ax.set_xticks(x)
ax.set_xticklabels(labels)
# Add a legend patch explaining the boxed success-count annotation
handles, labels_legend = ax.get_legend_handles_labels()
explain_patch = Patch(facecolor='white', edgecolor='black', label='xx settled out of yy attempts', alpha=0.95)
handle_map = dict(zip(labels_legend, handles))
ordered_labels = [
    'PX4 - Y-axis',
    'PX4 - Euclidean',
    'NMPC mocap - Y-axis',
    'NMPC mocap - Euclidean',
    'NMPC UKF - Y-axis',
    'NMPC UKF - Euclidean',
]
ordered_handles = [handle_map[label] for label in ordered_labels]
ordered_handles.append(explain_patch)
ordered_labels.append('xx settled out of yy attempts')
ax.legend(ordered_handles, ordered_labels, loc='upper center', fontsize=14, bbox_to_anchor=(0.5, -0.15), ncol=4)
plt.tight_layout()
plt.savefig('figures/full_system_settling_times_comparison.png', dpi=300)
plt.show()