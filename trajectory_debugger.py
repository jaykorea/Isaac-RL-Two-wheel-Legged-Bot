import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the provided CSV file
file_path = 'dataset/trajectory/dataset_240708_step108.csv'
data = pd.read_csv(file_path)

ctrl_hz = 50

# Set font size smaller
plt.rcParams.update({'font.size': 8.5})

# List of axes to plot (excluding the non-existent positions for wheels)
joint_names = ['left_hip', 'right_hip', 'left_shoulder', 'right_shoulder', 'left_leg', 'right_leg']
wheel_names = ['left_wheel', 'right_wheel']

# Time axis (assuming data index represents time steps and should be converted to seconds)
time = (data.index + 1) / ctrl_hz

# Create a figure
fig = plt.figure(figsize=(18, 12))

# Plot joint positions
for i, joint_name in enumerate(joint_names):
    plt.subplot(4, 2, i + 1)
    plt.plot(time, data[f'{joint_name}_current_position'], label=f'Current {joint_name.capitalize()} Pos', color='blue', marker='o', markersize=1, linewidth=1)
    plt.plot(time, data[f'{joint_name}_target_position'], label=f'Target {joint_name.capitalize()} Pos', color='red', marker='o', markersize=1, linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Position (rad)')
    plt.title(f'{joint_name.capitalize()} Positions', loc='left')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.legend()

# Plot wheel velocities
for i, wheel_name in enumerate(wheel_names):
    plt.subplot(4, 2, 6 + i + 1)
    plt.plot(time, data[f'{wheel_name}_current_velocity'], label=f'Current {wheel_name.capitalize()} Vel', color='blue', marker='o', markersize=1, linewidth=1)
    plt.plot(time, data[f'{wheel_name}_target_velocity'], label=f'Target {wheel_name.capitalize()} Vel', color='red', marker='o', markersize=1, linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (rad/s)')
    plt.title(f'{wheel_name.capitalize()} Velocities', loc='left')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.legend()

# Adjust layout with specific hspace
plt.tight_layout()
plt.subplots_adjust(hspace=0.21)
plt.show()

# Create a new figure for torques
fig_torque = plt.figure(figsize=(18, 12))

# Plot joint torques
for i, joint_name in enumerate(joint_names):
    plt.subplot(4, 2, i + 1)
    plt.plot(time, data[f'{joint_name}_torque'], label=f'Current {joint_name.capitalize()} Torque', color='orange', marker='o', markersize=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (Nm)')
    plt.title(f'{joint_name.capitalize()} Torques', loc='left')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.legend()

for i, wheel_name in enumerate(wheel_names):
    plt.subplot(4, 2, 6 + i + 1)
    plt.plot(time, data[f'{wheel_name}_torque'], label=f'Current {wheel_name.capitalize()} Torque', color='orange', marker='o', markersize=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (Nm)')
    plt.title(f'{wheel_name.capitalize()} Torques', loc='left')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.legend()

# Adjust layout with specific hspace
plt.tight_layout()
plt.subplots_adjust(hspace=0.21)
plt.show()