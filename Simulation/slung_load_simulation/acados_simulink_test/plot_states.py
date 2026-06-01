import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


with h5py.File('sim_data.mat', 'r') as f:
    data = np.array(f['data'])

with h5py.File('sim_control_output.mat', 'r') as f:
    control_data = np.array(f['data'])


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

thrust = control_data[:,1]
torque = control_data[:,2:5]

drone_pose = load_pose - cable_vec * 3.0

# Plot setpoint (2, -2, 0.75) as dashed line
fig, axs = plt.subplots(3, 2, figsize=(15, 9), sharex=True)
axs[0,0].plot(time, load_pose[:,0], label='Load X Position', color='tab:blue')
axs[0,0].axhline(2.0, color='r', linestyle='--', label='Setpoint X=2m')
axs[0,0].set_ylabel('X Position (m)')
axs[0,0].legend()
axs[0,0].grid()
axs[0,0].set_title('Load Position Over Time', fontsize=14)
axs[1,0].plot(time, load_pose[:,1], label='Load Y Position', color='tab:orange')
axs[1,0].axhline(-2.0, color='r', linestyle='--', label='Setpoint Y=-2m')
axs[1,0].set_ylabel('Y Position (m)')
axs[1,0].legend()
axs[1,0].grid()
axs[2,0].plot(time, load_pose[:,2], label='Load Z Position', color='tab:green')
axs[2,0].axhline(0.75, color='r', linestyle='--', label='Setpoint Z=0.75m')
axs[2,0].set_ylabel('Z Position (m)')
axs[2,0].set_xlabel('Time (s)')
axs[2,0].legend()
axs[2,0].grid()

# Plot load velocity in individual 2d plots, ignore drone velocity
axs[0,1].plot(time, load_vel[:,0], label='Load X Velocity')
axs[0,1].plot(time, load_vel[:,1], label='Load Y Velocity')
axs[0,1].plot(time, load_vel[:,2], label='Load Z Velocity')
axs[0,1].set_ylabel('Load Velocity (m/s)')
axs[0,1].legend()
axs[0,1].grid()
axs[0,1].set_title('Load Velocity Over Time', fontsize=14)

# Plot cable vector components
axs[1,1].plot(time, cable_vec[:,0], label='Cable X Component')
axs[1,1].plot(time, cable_vec[:,1], label='Cable Y Component')
axs[1,1].plot(time, cable_vec[:,2], label='Cable Z Component')
axs[1,1].set_ylabel('Normalized Vector Components')
axs[1,1].legend()
axs[1,1].grid()
axs[1,1].set_title('Cable Vector Components Over Time', fontsize=14)

# Plot angular velocity of the payload vector
axs[2,1].plot(time, w_cable_vec[:,0], label='Angular Vel X')
axs[2,1].plot(time, w_cable_vec[:,1], label='Angular Vel Y')
axs[2,1].plot(time, w_cable_vec[:,2], label='Angular Vel Z')
axs[2,1].set_ylabel('Angular Velocity (rad/s)')
axs[2,1].set_xlabel('Time (s)')
axs[2,1].legend()
axs[2,1].grid()
axs[2,1].set_title('Angular Velocity of Payload Vector Over Time', fontsize=14)
plt.tight_layout()
plt.savefig('figures/states.png', dpi=300)
plt.show()


# Make a plot of the commanded thrust and angular torques
fig, axs = plt.subplots(1, 2, figsize=(15, 3), sharex=True)
axs[0].plot(time, thrust, label='Thrust Command')
axs[0].set_ylabel('Thrust (N)')
axs[0].legend()
axs[0].grid()
axs[0].set_title('Thrust Inputs Over Time', fontsize=14)
axs[0].set_xlabel('Time (s)')
axs[1].plot(time, torque[:,0], label='Torque X Command')
axs[1].set_ylabel('Torque (Nm)')
axs[1].plot(time, torque[:,1], label='Torque Y Command')
axs[1].plot(time, torque[:,2], label='Torque Z Command')
axs[1].set_xlabel('Time (s)')
axs[1].set_title('Torque Inputs Over Time', fontsize=14)
axs[1].legend()
axs[1].grid()
plt.tight_layout()
plt.savefig('figures/control_inputs.png', dpi=300)
plt.show()

