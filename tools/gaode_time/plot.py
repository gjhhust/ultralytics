import matplotlib.pyplot as plt
import matplotlib.font_manager as fm # Still import in case default sans-serif works
import numpy as np
import os
import matplotlib

# Print cache directory for user reference (optional, can be removed)
# print("Matplotlib cache directory:", matplotlib.get_cachedir())

# --- Data preparation ---
# Extract AP75 and time(ms) data from tables
# Data structure: { 'quantization_mode': { 'model_name': (time_ms, AP75) } }

# Table 1 ONNX CPU (FP32 baseline)
data_onnx_cpu_fp32 = {
    'yolov8s': (245, 37.82),
    'yolov8l': (343, 40.68),
    'yoloft-s': (290, 45.96),
    'yoloft-l': (352, 52.90),
    'yoloft-x': (398, 58.30)
}

# Table 2 BModel FP16 (Assuming b16 is FP16)
data_bmodel_fp16 = {
    'yolov8s': (38.6, 26.22),
    'yolov8l': (56.2, 33.38),
    'yoloft-s': (45.6, 36.66),
    'yoloft-l': (70.3, 42.98),
    'yoloft-x': (101.4, 41.30)
}

# Table 3 OpenVINO INT8 CPU (INT8 test proxy)
# Note: yoloft-s INT8 data is missing
data_openvino_int8_cpu = {
    'yolov8s': (133, 36.78),
    'yolov8l': (240, 40.16),
    'yoloft-l': (168, 50.82),
    'yoloft-x': (258, 57.38)
}

# Consolidate all data points by mode
all_data_by_mode = {
    'ONNX CPU (FP32)': {'data': data_onnx_cpu_fp32, 'color': 'blue', 'marker': 'o'},
    'BModel FP16': {'data': data_bmodel_fp16, 'color': 'orange', 'marker': 's'},
    'OpenVINO INT8 CPU': {'data': data_openvino_int8_cpu, 'color': 'green', 'marker': '^'}
}

# Model name mapping and consistent markers across modes
model_markers = {
    'yolov8s': 'o',  # circle
    'yolov8l': 's',  # square
    'yoloft-s': '^', # triangle_up
    'yoloft-l': 'D', # diamond
    'yoloft-x': 'X'  # x
}

model_names_map = {
    'yolov8s': 'YOLOv8-s',
    'yolov8l': 'YOLOv8-l',
    'yoloft-s': 'YOLOFT-s',
    'yoloft-l': 'YOLOFT-l',
    'yoloft-x': 'YOLOFT-x'
}

# --- Organize data by model for plotting lines ---
data_by_model = {model_name: [] for model_name in model_names_map.keys()}

for mode, mode_info in all_data_by_mode.items():
    for model_name, (time, ap75) in mode_info['data'].items():
        data_by_model[model_name].append({'time': time, 'ap75': ap75, 'mode': mode}) # Store all relevant info

# --- Plotting ---
plt.figure(figsize=(12, 8)) # Set figure size

# --- Plot Lines Connecting Model Series ---
# Define line styles and colors for each model series
model_line_styles = {
    'yolov8s': '-',   # solid line
    'yolov8l': '--',  # dashed line
    'yoloft-s': '-.', # dash-dot line
    'yoloft-l': ':',  # dotted line
    'yoloft-x': '-',  # solid line (can use color to differentiate if styles repeat)
}
model_line_colors = {
    'yolov8s': 'gray',
    'yolov8l': 'dimgray',
    'yoloft-s': 'purple',
    'yoloft-l': 'red', # Highlight YOLOFT-l
    'yoloft-x': 'darkviolet',
}

legend_handles_model_lines = []
legend_labels_model_lines = []

for model_name, points in data_by_model.items():
    if len(points) > 1: # Only draw a line if there's more than one point for this model
        # Sort points by time for connecting lines correctly
        sorted_points = sorted(points, key=lambda p: p['time'])
        times = [p['time'] for p in sorted_points]
        ap75s = [p['ap75'] for p in sorted_points]

        # Plot the line
        line_style = model_line_styles.get(model_name, '-') # Default to solid
        line_color = model_line_colors.get(model_name, 'black') # Default to black
        line, = plt.plot(times, ap75s, linestyle=line_style, color=line_color, marker=None, linewidth=2, zorder=3) # zorder below points, capture line artist

        # Add line legend entry
        legend_handles_model_lines.append(line)
        legend_labels_model_lines.append(model_names_map[model_name])

# --- Plot Points (Color by Mode, Marker by Model) ---
# Create proxy artists for mode legend (using the first model's marker for that mode's color)
legend_handles_mode = []
legend_labels_mode = []
used_mode_colors = set() # Prevent adding duplicate mode colors if modes share colors (not the case here, but good practice)

for mode, mode_info in all_data_by_mode.items():
     if mode_info['color'] not in used_mode_colors:
        # Find a sample marker for this color (e.g., the marker for the first model in this mode)
        first_model_name_in_mode = list(mode_info['data'].keys())[0] if mode_info['data'] else 'yolov8s' # Use yolov8s marker as default if mode has no data
        marker_for_legend = model_markers.get(first_model_name_in_mode, 'o') # Get marker, default to circle

        legend_handles_mode.append(plt.Line2D([0], [0], linestyle='none', marker=marker_for_legend, color=mode_info['color'], markersize=10))
        legend_labels_mode.append(mode)
        used_mode_colors.add(mode_info['color'])

     # Now plot the actual points
     for model_name, (time, ap75) in mode_info['data'].items():
        marker = model_markers.get(model_name, 'o') # Get model-specific marker
        plt.scatter(time, ap75, color=mode_info['color'], marker=marker, s=150, zorder=5) # s control size, zorder ensures points are above lines


# --- Add Legends ---
# Plotting order matters for legend placement
# First, add the mode legend
legend1 = plt.legend(handles=legend_handles_mode, labels=legend_labels_mode, title='Quantization Mode (Color)', loc='upper right')
plt.gca().add_artist(legend1) # Add the first legend to the axes

# Then, add the model series line legend
legend2 = plt.legend(handles=legend_handles_model_lines, labels=legend_labels_model_lines, title='Model Series (Trajectory)', loc='lower left')

# Optional: Add a third legend for the model markers if needed, but lines/colors should be sufficient
# legend_handles_model_points = []
# legend_labels_model_points = []
# used_model_markers_legend = set()
# for model_name, marker in model_markers.items():
#     if marker not in used_model_markers_legend:
#         legend_handles_model_points.append(plt.Line2D([0], [0], linestyle='none', marker=marker, color='black', markersize=10))
#         legend_labels_model_points.append(model_names_map[model_name])
#         used_model_markers_legend.add(marker)
# legend3 = plt.legend(handles=legend_handles_model_points, labels=legend_labels_model_points, title='Model Name (Marker)', loc='lower right')


# --- Set Titles and Labels (in English) ---
plt.title('Model Quantization Performance (AP75 vs Inference Time)') # Plot title
plt.xlabel('Inference Time (ms)') # X-axis label
plt.ylabel('AP75 (%)')     # Y-axis label
plt.grid(True, linestyle='--', alpha=0.6) # Add grid lines

# --- Set Axis Limits ---
all_times = [p['time'] for model_points in data_by_model.values() for p in model_points]
all_ap75s = [p['ap75'] for model_points in data_by_model.values() for p in model_points]


if all_times and all_ap75s:
    max_time = max(all_times) if all_times else 100
    max_ap75 = max(all_ap75s) if all_ap75s else 100
    min_ap75 = min(all_ap75s) if all_ap75s else 0

    plt.xlim(0, max_time * 1.1) # x-axis starts from 0, 10% margin
    # y-axis starts from 0 or slightly below min if min > 0, 10% margin above max
    plt.ylim(min_ap75 * 0.95 if min_ap75 > 0 else 0, max_ap75 * 1.1)
else:
     print("No data available to set axis limits.")


plt.tight_layout() # Automatically adjust layout
plt.savefig('quantization_performance_pareto_english.png', dpi=300) # Save the plot with an English name
# plt.show() # Uncomment to show the plot window interactively