import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np


def get_group(name):
    if name.startswith("A-"):
        return "A"
    elif name.startswith("M-"):
        return "M"
    elif name.startswith("P-"):
        return "P"
    return "Other"

cd = os.getcwd()
directory = os.path.join(cd, 'result')
output_directory = os.path.join(directory, 'stats')
os.makedirs(output_directory, exist_ok=True)

diff_data = {}
int_diff_data = {}
time_data = {}
avg_diff_data = {}

for filename in os.listdir(directory):
    if filename.endswith(".json"):
        filepath = os.path.join(directory, filename)
        filename = filename.split('_')[-1].split('.')[0]
        with open(filepath, "r") as file:
            data = json.load(file)
        
        diffs = [entry["diff"] for entry in data.values()]
        avg_diff = np.mean(diffs)
        diff_data[filename] = avg_diff
        
        time_groups = {"A": [], "M": [], "P": []}
        group_diff = {"A": [], "M": [], "P": []}
        
        for key, entry in data.items():
            group = get_group(key)
            if group in time_groups:
                time_groups[group].append(entry["time"])
                group_diff[group].append(entry["diff"])
                
        
        avg_times = {group: np.mean(times) if times else 0 for group, times in time_groups.items()}
        group_avg_diff = {group: np.mean(diff) if diff else 0 for group, diff in group_diff.items()}
        time_data[filename] = avg_times
        avg_diff_data[filename] = group_avg_diff
        


plt.figure(figsize=(10, 5))
plt.bar(diff_data.keys(), diff_data.values(), color=['r', 'g', 'b'])
plt.xlabel("метод")
plt.ylabel("Средний diff")
plt.title("Среднее значение diff по методам")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7) 
for i, v in enumerate(diff_data.values()):
    plt.text(i, v + 0.1, f"{v:.2f}", ha='center')
plt.savefig(os.path.join(output_directory, "diff_plot.png"))
plt.close()

labels = ["A", "M", "P"]
x = np.arange(len(labels))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 5))
ax.grid(axis='y', linestyle='--', alpha=0.7)

for i, (filename, times) in enumerate(avg_diff_data.items()):
    values = [times[label] for label in labels]
    ax.bar(x + i * width, values, width, label=filename)
    for j, v in enumerate(values):
        ax.text(x[j] + i * width, v + 5, f"{v:.1f}", ha='center')

ax.set_xlabel("Группы")
ax.set_ylabel("Средний diff")
ax.set_title("Средний diff по группам по файлам")
ax.set_xticks(x + width / 2)
ax.set_xticklabels(labels)
ax.legend()
plt.savefig(os.path.join(output_directory, "group_avg_diff_plot.png"))
plt.close()

labels = ["A", "M", "P"]
x = np.arange(len(labels))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 5))
ax.grid(axis='y', linestyle='--', alpha=0.7)

for i, (filename, times) in enumerate(time_data.items()):
    values = [times[label] for label in labels]
    ax.bar(x + i * width, values, width, label=filename)
    for j, v in enumerate(values):
        ax.text(x[j] + i * width, v + 5, f"{v:.1f}", ha='center')

ax.set_xlabel("Группы")
ax.set_ylabel("Среднее время")
ax.set_title("Среднее время выполнения по группам")
ax.set_xticks(x + width / 2)
ax.set_xticklabels(labels)
ax.legend()
plt.savefig(os.path.join(output_directory, "time_plot.png"))
plt.close()
