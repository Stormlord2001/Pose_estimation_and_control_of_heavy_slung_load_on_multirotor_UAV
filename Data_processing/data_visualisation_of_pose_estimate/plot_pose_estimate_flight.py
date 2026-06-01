import csv
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import numpy as np

# ---- CONFIG ----
#path = "/home/rasmus-storm/Desktop/tests/14_04_bias_estimation/video_20/dataset"
path = "/home/rasmus-storm/Desktop/tests/05_05/handin_test_folder/dataset_world"
csv_path = f"{path}.csv"

fs = 30.0  # Hz
dt = 1.0 / fs

# ---- LOAD CSV ----
p_L_W_est = []
q_L_W_est = []
p_L_W_mocap = []
q_L_W_mocap = []

with open(csv_path, newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        p_L_W_est.append([float(row[0]), float(row[1]), float(row[2])])
        q_L_W_est.append([float(row[3]), float(row[4]), float(row[5]), float(row[6])])

        p_L_W_mocap.append([float(row[7]), float(row[8]), float(row[9])])
        q_L_W_mocap.append([float(row[10]), float(row[11]), float(row[12]), float(row[13])])

# ---- SKIP FRAMES ----
skip_frames = 0#950+3495 #3603#1400
cut_frames = 1#len(p_L_W_mocap) - skip_frames - 300
Allign_frames = 0#1500

p_L_W_est = p_L_W_est[skip_frames:-cut_frames-Allign_frames]
q_L_W_est = q_L_W_est[skip_frames:-cut_frames-Allign_frames]
p_L_W_mocap = p_L_W_mocap[skip_frames:-cut_frames-Allign_frames]
q_L_W_mocap = q_L_W_mocap[skip_frames:-cut_frames-Allign_frames]

# ---- RMSE FUNCTION ----
def rmse(gt, est):
    return math.sqrt(sum((g - e) ** 2 for g, e in zip(gt, est)) / len(gt))

# ---- TRANSLATION RMSE ----
rmse_x = rmse([p[0] for p in p_L_W_mocap], [p[0] for p in p_L_W_est])
rmse_y = rmse([p[1] for p in p_L_W_mocap], [p[1] for p in p_L_W_est])
rmse_z = rmse([p[2] for p in p_L_W_mocap], [p[2] for p in p_L_W_est])

# ---- ROTATION ERROR (ANGLE) ----
R_gt = [R.from_quat(q) for q in q_L_W_mocap]
R_est = [R.from_quat(q) for q in q_L_W_est]

angle_err_deg = []

for r_gt, r_est in zip(R_gt, R_est):
    r_err = r_gt.inv() * r_est
    q_err = r_err.as_quat()

    w = np.clip(abs(q_err[3]), -1.0, 1.0)
    angle_err_deg.append(math.degrees(2 * math.acos(w)))

mean_rot_angle = np.mean(angle_err_deg)

print(f"Rotation mean error (angle): {mean_rot_angle:.4f} deg")

# ---- ALIGN QUATERNIONS ----
q_gt = np.array(q_L_W_mocap)
q_est = np.array(q_L_W_est)

for i in range(len(q_gt)):
    if np.dot(q_gt[i], q_est[i]) < 0:
        q_est[i] = -q_est[i]

qx_gt, qy_gt, qz_gt, qw_gt = q_gt.T
qx_est, qy_est, qz_est, qw_est = q_est.T

# ---- QUATERNION RMSE ----
rmse_qx = rmse(qx_gt, qx_est)
rmse_qy = rmse(qy_gt, qy_est)
rmse_qz = rmse(qz_gt, qz_est)
rmse_qw = rmse(qw_gt, qw_est)

print(f"RMSE X: {rmse_x:.4f}")
print(f"RMSE Y: {rmse_y:.4f}")
print(f"RMSE Z: {rmse_z:.4f}")
print(f"Rotation mean (angle): {mean_rot_angle:.4f} deg")

print(f"Quaternion RMSE qx: {rmse_qx:.6f}")
print(f"Quaternion RMSE qy: {rmse_qy:.6f}")
print(f"Quaternion RMSE qz: {rmse_qz:.6f}")
print(f"Quaternion RMSE qw: {rmse_qw:.6f}")

# ---- TIME AXIS (SECONDS) ----
n = len(p_L_W_mocap)
time = np.arange(n) * dt

# ---- PLOTTING (4x2 GRID) ----
fig, axes = plt.subplots(4, 2, figsize=(15, 9), sharex=True)

# LEFT: translation + angle error
left_plots = [
    ([p[0] for p in p_L_W_mocap], [p[0] for p in p_L_W_est], "X (m)", rmse_x),
    ([p[1] for p in p_L_W_mocap], [p[1] for p in p_L_W_est], "Y (m)", rmse_y),
    ([p[2] for p in p_L_W_mocap], [p[2] for p in p_L_W_est], "Z (m)", rmse_z),
]


y_min_max = [min(min(p[0] for p in p_L_W_mocap), min(p[0] for p in p_L_W_est)), max(max(p[0] for p in p_L_W_mocap), max(p[0] for p in p_L_W_est)),
             min(min(p[1] for p in p_L_W_mocap), min(p[1] for p in p_L_W_est)), max(max(p[1] for p in p_L_W_mocap), max(p[1] for p in p_L_W_est)),
             min(min(p[2] for p in p_L_W_mocap), min(p[2] for p in p_L_W_est)), max(max(p[2] for p in p_L_W_mocap), max(p[2] for p in p_L_W_est))]
biggest_range = max(y_min_max[1] - y_min_max[0], y_min_max[3] - y_min_max[2], y_min_max[5] - y_min_max[4]) * 1.1
y_center = [(y_min_max[0] + y_min_max[1]) / 2, (y_min_max[2] + y_min_max[3]) / 2, (y_min_max[4] + y_min_max[5]) / 2]
y_limits = [(y_center[0] - biggest_range / 2, y_center[0] + biggest_range / 2), (y_center[1] - biggest_range / 2, y_center[1] + biggest_range / 2), (y_center[2] - biggest_range / 2, y_center[2] + biggest_range / 2)]

for i, (gt, est, label, error) in enumerate(left_plots):
    ax = axes[i, 0]
    ax.plot(time, gt, label="GT")
    ax.plot(time, est, label="Est", alpha=0.7)
    ax.set_title(f"{label} (RMSE={error:.4f}m)")
    ax.set_ylabel(label)
    ax.grid(True)
    ax.set_ylim(y_limits[i])
    ax.legend()

axes[3, 0].plot(time, angle_err_deg)
axes[3, 0].set_title(f"Rotation Error Angle (mean={mean_rot_angle:.4f} deg)")
axes[3, 0].set_ylabel("deg")
axes[3, 0].grid(True)

# RIGHT: quaternion components
quat_data = [
    (qx_gt, qx_est, "qx", rmse_qx),
    (qy_gt, qy_est, "qy", rmse_qy),
    (qz_gt, qz_est, "qz", rmse_qz),
    (qw_gt, qw_est, "qw", rmse_qw),
]

quat_min_max = [min(min(qx_gt), min(qx_est)), max(max(qx_gt), max(qx_est)),
                min(min(qy_gt), min(qy_est)), max(max(qy_gt), max(qy_est)),
                min(min(qz_gt), min(qz_est)), max(max(qz_gt), max(qz_est)),
                min(min(qw_gt), min(qw_est)), max(max(qw_gt), max(qw_est))]
quat_biggest_range = max(quat_min_max[1] - quat_min_max[0], quat_min_max[3] - quat_min_max[2], quat_min_max[5] - quat_min_max[4])*1.1
quat_4_range = (quat_min_max[7] - quat_min_max[6])*1.1
quat_y_center = [(quat_min_max[0] + quat_min_max[1]) / 2, (quat_min_max[2] + quat_min_max[3]) / 2, (quat_min_max[4] + quat_min_max[5]) / 2, (quat_min_max[6] + quat_min_max[7]) / 2]
quat_y_limits = [(quat_y_center[0] - quat_biggest_range / 2, quat_y_center[0] + quat_biggest_range / 2),
                 (quat_y_center[1] - quat_biggest_range / 2, quat_y_center[1] + quat_biggest_range / 2),
                 (quat_y_center[2] - quat_biggest_range / 2, quat_y_center[2] + quat_biggest_range / 2),
                 (quat_y_center[3] - quat_4_range / 2, quat_y_center[3] + quat_4_range / 2)]

for i, (gt, est, label, error) in enumerate(quat_data):
    ax = axes[i, 1]
    ax.plot(time, gt, label="GT")
    ax.plot(time, est, label="Est", alpha=0.7)
    ax.set_title(f"{label} (RMSE={error:.5f})")
    ax.grid(True)
    ax.set_ylim(quat_y_limits[i])
    ax.legend()

# ---- FINAL TOUCH ----
for ax in axes[-1]:
    ax.set_xlabel("Time (s)")

plt.tight_layout()
plt.show()

fig.savefig(f"{path}.png", dpi=300)