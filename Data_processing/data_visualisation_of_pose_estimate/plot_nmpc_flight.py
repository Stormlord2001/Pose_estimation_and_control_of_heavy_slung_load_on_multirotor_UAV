import numpy as np
import matplotlib.pyplot as plt

path = "/home/rasmus-storm/Desktop/tests/14_04_bias_estimation/video_20/dataset_nmpc"
#path = "/home/rasmus-storm/Desktop/tests/05_05/rosbag_15/dataset_nmpc"
csv_path = f"{path}.csv"

data = np.loadtxt(csv_path, delimiter=',')
data = data.T  # columns

segment1 = data[:12]      # motion capture
segment2 = data[12:24]    # estimated
bias = np.squeeze(data[24])


# 05_05 rosbag_15: offset = 0, start_idx = 1318, end_idx = -40 start_idx + 480
# 14_04 video_20: offset = 0, start_idx = 1318 + 5468, end_idx = 1318 + 5468 + 480
offset = 0#-1235#6229#5098#3837#2579#1317  # <-- offset to align with flying data
start_idx = 2360 + offset #1318 #1318# + 5468#5528#2464 + offset #+ 122 #860 #4250      # <-- start sample
end_idx = -6010 + offset #start_idx + 480#-8400#-6230 + offset#1200 #6000     # <-- end sample (None = use full length)





if end_idx is None:
    end_idx = segment1.shape[1]

segment1 = segment1[:, start_idx:end_idx]
segment2 = segment2[:, start_idx:end_idx]
bias = bias[start_idx:end_idx]

print(f"Using samples from {start_idx} to {end_idx} (total {segment1.shape[1]} samples)")
# -----------------------
# Time base (50 Hz)
# -----------------------
fs = 50.0
dt = 1.0 / fs
N = segment1.shape[1]
t = np.arange(N) * dt

# -----------------------
# Normalize world vector (6:9)
# -----------------------
vec = segment1[6:9]
norm = np.linalg.norm(vec, axis=0)
norm[norm == 0] = 1e-8
segment1[6:9] = vec / norm

# -----------------------
# RMSE function
# -----------------------
def rmse(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.sqrt(np.mean((a - b) ** 2))

# -----------------------
# Compute RMSE (NO bias)
# -----------------------
rmse_vals = np.array([
    rmse(segment1[i], segment2[i]) for i in range(12)
])

# -----------------------
# Plot layout
# -----------------------
fig = plt.figure(figsize=(15, 14))
gs = fig.add_gridspec(5, 3)

def plot_pair(ax, i, title, ylabel, unit=""):
    ax.plot(t, segment1[i], label="Motion Capture", linewidth=1)
    ax.plot(t, segment2[i], label="Estimated", linewidth=1)

    ax.set_title(f"{title} | RMSE: {rmse_vals[i]:.4f}{unit}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()

axes = ["x", "y", "z"]
# -----------------------
# Position (m)
# -----------------------
y_min_max = [min(min(segment1[i]), min(segment2[i])) for i in range(3)] + [max(max(segment1[i]), max(segment2[i])) for i in range(3)]
biggest_range = max(y_min_max[3] - y_min_max[0], y_min_max[4] - y_min_max[1], y_min_max[5] - y_min_max[2]) * 1.1
y_center = [(y_min_max[0] + y_min_max[3]) / 2, (y_min_max[1] + y_min_max[4]) / 2, (y_min_max[2] + y_min_max[5]) / 2]
y_limits = [(y_center[0] - biggest_range / 2, y_center[0] + biggest_range / 2), (y_center[1] - biggest_range / 2, y_center[1] + biggest_range / 2), (y_center[2] - biggest_range / 2, y_center[2] + biggest_range / 2)]
for i in range(3):
    ax = fig.add_subplot(gs[0, i])
    plot_pair(ax, i, f"Position {axes[i]}-axis", "Position [m]", "m")
    ax.set_ylim(y_limits[i])

# -----------------------
# Velocity (m/s)
# -----------------------
y_min_max = [min(min(segment1[i]), min(segment2[i])) for i in range(3, 6)] + [max(max(segment1[i]), max(segment2[i])) for i in range(3, 6)]
biggest_range = max(y_min_max[3] - y_min_max[0], y_min_max[4] - y_min_max[1], y_min_max[5] - y_min_max[2]) * 1.1
y_center = [(y_min_max[0] + y_min_max[3]) / 2, (y_min_max[1] + y_min_max[4]) / 2, (y_min_max[2] + y_min_max[5]) / 2]
y_limits = [(y_center[0] - biggest_range / 2, y_center[0] + biggest_range / 2), (y_center[1] - biggest_range / 2, y_center[1] + biggest_range / 2), (y_center[2] - biggest_range / 2, y_center[2] + biggest_range / 2)]
for i in range(3, 6):
    ax = fig.add_subplot(gs[1, i - 3])
    plot_pair(ax, i, f"Velocity {axes[i-3]}-axis", "Velocity [m/s]", "m/s")
    ax.set_ylim(y_limits[i - 3])

# -----------------------
# World vector
# -----------------------
y_min_max = [min(min(segment1[i]), min(segment2[i])) for i in range(6, 9)] + [max(max(segment1[i]), max(segment2[i])) for i in range(6, 9)]
biggest_range = max(y_min_max[3] - y_min_max[0], y_min_max[4] - y_min_max[1], y_min_max[5] - y_min_max[2]) * 1.1
y_center = [(y_min_max[0] + y_min_max[3]) / 2, (y_min_max[1] + y_min_max[4]) / 2, (y_min_max[2] + y_min_max[5]) / 2]
y_limits = [(y_center[0] - biggest_range / 2, y_center[0] + biggest_range / 2), (y_center[1] - biggest_range / 2, y_center[1] + biggest_range / 2), (y_center[2] - biggest_range / 2, y_center[2] + biggest_range / 2)]
for i in range(6, 9):
    ax = fig.add_subplot(gs[2, i - 6])
    plot_pair(ax, i, f"World Vector {axes[i-6]}-axis", "Normalized direction")
    ax.set_ylim(y_limits[i - 6])

# -----------------------
# Angular velocity (rad/s)
# -----------------------
y_min_max = [min(min(segment1[i]), min(segment2[i])) for i in range(9, 12)] + [max(max(segment1[i]), max(segment2[i])) for i in range(9, 12)]
biggest_range = max(y_min_max[3] - y_min_max[0], y_min_max[4] - y_min_max[1], y_min_max[5] - y_min_max[2]) * 1.1
y_center = [(y_min_max[0] + y_min_max[3]) / 2, (y_min_max[1] + y_min_max[4]) / 2, (y_min_max[2] + y_min_max[5]) / 2]
y_limits = [(y_center[0] - biggest_range / 2, y_center[0] + biggest_range / 2), (y_center[1] - biggest_range / 2, y_center[1] + biggest_range / 2), (y_center[2] - biggest_range / 2, y_center[2] + biggest_range / 2)]
for i in range(9, 12):
    ax = fig.add_subplot(gs[3, i - 9])
    plot_pair(ax, i, f"Angular Velocity {axes[i-9]}-axis", "Angular velocity [rad/s]", "rad/s")
    ax.set_ylim(y_limits[i - 9])

# -----------------------
# Bias (no RMSE)
# -----------------------
ax_bias = fig.add_subplot(gs[4, :])
ax_bias.plot(t, bias, label="Acceleration Bias", linewidth=2)

ax_bias.set_title("Estimated Acceleration Bias (m/s²)")
ax_bias.set_xlabel("Time [s]")
ax_bias.set_ylabel("Acceleration [m/s²]")
ax_bias.grid(True)
ax_bias.legend()

# -----------------------
# Final layout
# -----------------------
plt.tight_layout()
plt.savefig(f"{path}_comparison.png", dpi=300)
plt.show()