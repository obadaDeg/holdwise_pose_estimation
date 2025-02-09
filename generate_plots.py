import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the datasets
pose_estimation_approach = pd.read_csv("Training_Approach_1.csv")
multimodal_approach = pd.read_csv("Training_Approach_2.csv")

# Extracting relevant data for Approach 1
epochs_1 = pose_estimation_approach["Epoch"]
accuracy_1 = pose_estimation_approach["Accuracy"]
val_accuracy_1 = pose_estimation_approach["Val Accuracy"]
loss_1 = pose_estimation_approach["Loss"]
val_loss_1 = pose_estimation_approach["Val Loss"]

# Extracting relevant data for Approach 2
epochs_2 = multimodal_approach["Epoch"]
accuracy_2 = multimodal_approach["Accuracy"]
val_accuracy_2 = multimodal_approach["Val Accuracy"]
loss_2 = multimodal_approach["Loss"]
val_loss_2 = multimodal_approach["Val Loss"]

# Enhanced Visualization
fig, axes = plt.subplots(figsize=(10, 5))

# Training Loss
axes.plot(epochs_1, loss_1, label="Pose Estimation", color="blue")
axes.plot(epochs_2, loss_2, label="Multimodal", color="red")
axes.set_title("Training Loss Comparison")
axes.set_xlabel("Epochs")
axes.set_ylabel("Loss")
axes.legend()
axes.grid()
plt.savefig("output/training_loss_comparison.png")

fig, axes = plt.subplots(figsize=(10, 5))
# Validation Loss
axes.plot(epochs_1, val_loss_1, label="Pose Estimation", color="blue", linestyle="dashed")
axes.plot(epochs_2, val_loss_2, label="Multimodal", color="red", linestyle="dashed")
axes.set_title("Validation Loss Comparison")
axes.set_xlabel("Epochs")
axes.set_ylabel("Loss")
axes.legend()
axes.grid()
plt.savefig("output/validation_loss_comparison.png")

fig, axes = plt.subplots(figsize=(10, 5))
# Training Accuracy
axes.plot(epochs_1, accuracy_1, label="Pose Estimation", color="blue")
axes.plot(epochs_2, accuracy_2, label="Multimodal", color="red")
axes.set_title("Training Accuracy Comparison")
axes.set_xlabel("Epochs")
axes.set_ylabel("Accuracy")
axes.legend()
axes.grid()
plt.savefig("output/training_accuracy_comparison.png")

fig, axes = plt.subplots(figsize=(10, 5))
# Validation Accuracy
axes.plot(epochs_1, val_accuracy_1, label="Pose Estimation", color="blue", linestyle="dashed")
axes.plot(epochs_2, val_accuracy_2, label="Multimodal", color="red", linestyle="dashed")
axes.set_title("Validation Accuracy Comparison")
axes.set_xlabel("Epochs")
axes.set_ylabel("Accuracy")
axes.legend()
axes.grid()

plt.tight_layout()
plt.savefig("output/training_comparison.png")

# Additional Insightful Plots
fig, axes = plt.subplots(figsize=(14, 10))

fig, axes = plt.subplots(figsize=(10, 5))
# Accuracy Difference Over Epochs
accuracy_diff = accuracy_1 - accuracy_2
axes.plot(epochs_1, accuracy_diff, label="Accuracy Difference (Pose - Multimodal)", color="green")
axes.set_title("Accuracy Difference Over Epochs")
axes.set_xlabel("Epochs")
axes.set_ylabel("Accuracy Difference")
axes.axhline(0, color="black", linestyle="dashed")
axes.legend()
axes.grid()
plt.savefig("output/accuracy_difference.png")

fig, axes = plt.subplots(figsize=(10, 5))
# Loss Difference Over Epochs
loss_diff = loss_1 - loss_2
axes.plot(epochs_1, loss_diff, label="Loss Difference (Pose - Multimodal)", color="purple")
axes.set_title("Loss Difference Over Epochs")
axes.set_xlabel("Epochs")
axes.set_ylabel("Loss Difference")
axes.axhline(0, color="black", linestyle="dashed")
axes.legend()
axes.grid()
plt.savefig("output/loss_difference.png")

fig, axes = plt.subplots(figsize=(10, 5))
# Performance Efficiency (Accuracy/Loss)
performance_ratio_1 = accuracy_1 / (loss_1 + 1e-8)  # Avoid division by zero
performance_ratio_2 = accuracy_2 / (loss_2 + 1e-8)
axes.plot(epochs_1, performance_ratio_1, label="Performance Ratio (Pose)", color="blue")
axes.plot(epochs_2, performance_ratio_2, label="Performance Ratio (Multimodal)", color="red")
axes.set_title("Performance Efficiency (Accuracy/Loss)")
axes.set_xlabel("Epochs")
axes.set_ylabel("Performance Ratio")
axes.legend()
axes.grid()
plt.savefig("output/performance_efficiency.png")

fig, axes = plt.subplots(figsize=(10, 5))
# Validation Accuracy Stability (Rate of Change in Validation Accuracy)
val_acc_change_1 = np.abs(np.diff(val_accuracy_1, prepend=val_accuracy_1.iloc[0]))
val_acc_change_2 = np.abs(np.diff(val_accuracy_2, prepend=val_accuracy_2.iloc[0]))
axes.plot(epochs_1, val_acc_change_1, label="Stability (Pose)", color="blue")
axes.plot(epochs_2, val_acc_change_2, label="Stability (Multimodal)", color="red")
axes.set_title("Validation Accuracy Stability")
axes.set_xlabel("Epochs")
axes.set_ylabel("Rate of Change")
axes.legend()
axes.grid()

plt.tight_layout()
plt.savefig("output/validation_accuracy_stability.png")

import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import confusion_matrix
import numpy as np

# Applying rolling average for smoothing (window size = 5 epochs)
window_size = 5

accuracy_1_smooth = accuracy_1.rolling(window=window_size, min_periods=1).mean()
val_accuracy_1_smooth = val_accuracy_1.rolling(window=window_size, min_periods=1).mean()
accuracy_2_smooth = accuracy_2.rolling(window=window_size, min_periods=1).mean()
val_accuracy_2_smooth = val_accuracy_2.rolling(window=window_size, min_periods=1).mean()

loss_1_smooth = loss_1.rolling(window=window_size, min_periods=1).mean()
val_loss_1_smooth = val_loss_1.rolling(window=window_size, min_periods=1).mean()
loss_2_smooth = loss_2.rolling(window=window_size, min_periods=1).mean()
val_loss_2_smooth = val_loss_2.rolling(window=window_size, min_periods=1).mean()

# Plot Smoothed Accuracy and Loss
fig, axes = plt.subplots(figsize=(10, 5))

# Smoothed Training Accuracy
axes.plot(epochs_1, accuracy_1_smooth, label="Pose Estimation", color="blue")
axes.plot(epochs_2, accuracy_2_smooth, label="Multimodal", color="red")
axes.set_title("Smoothed Training Accuracy")
axes.set_xlabel("Epochs")
axes.set_ylabel("Accuracy")
axes.legend()
axes.grid()
plt.savefig("output/smoothed_training_accuracy.png")

fig, axes = plt.subplots(figsize=(10, 5))
# Smoothed Validation Accuracy
axes.plot(epochs_1, val_accuracy_1_smooth, label="Pose Estimation", color="blue", linestyle="dashed")
axes.plot(epochs_2, val_accuracy_2_smooth, label="Multimodal", color="red", linestyle="dashed")
axes.set_title("Smoothed Validation Accuracy")
axes.set_xlabel("Epochs")
axes.set_ylabel("Accuracy")
axes.legend()
axes.grid()
plt.savefig("output/smoothed_validation_accuracy.png")

fig, axes = plt.subplots(figsize=(10, 5))
# Smoothed Training Loss
axes.plot(epochs_1, loss_1_smooth, label="Pose Estimation", color="blue")
axes.plot(epochs_2, loss_2_smooth, label="Multimodal", color="red")
axes.set_title("Smoothed Training Loss")
axes.set_xlabel("Epochs")
axes.set_ylabel("Loss")
axes.legend()
axes.grid()

plt.savefig("output/smoothed_training_loss.png")

fig, axes = plt.subplots(figsize=(10, 5))
# Smoothed Validation Loss
axes.plot(epochs_1, val_loss_1_smooth, label="Pose Estimation", color="blue", linestyle="dashed")
axes.plot(epochs_2, val_loss_2_smooth, label="Multimodal", color="red", linestyle="dashed")
axes.set_title("Smoothed Validation Loss")
axes.set_xlabel("Epochs")
axes.set_ylabel("Loss")
axes.legend()
axes.grid()

plt.tight_layout()
plt.savefig("output/smoothed_validation_loss.png")

# Plot Training vs Validation Accuracy Gap
fig, ax = plt.subplots(figsize=(10, 5))

accuracy_gap_1 = accuracy_1 - val_accuracy_1
accuracy_gap_2 = accuracy_2 - val_accuracy_2

ax.plot(epochs_1, accuracy_gap_1, label="Pose Estimation", color="blue")
ax.plot(epochs_2, accuracy_gap_2, label="Multimodal", color="red")
ax.set_title("Training vs Validation Accuracy Gap")
ax.set_xlabel("Epochs")
ax.set_ylabel("Accuracy Gap")
ax.axhline(0, color="black", linestyle="dashed")
ax.legend()
ax.grid()

plt.savefig("output/accuracy_gap.png")

# Plot Loss Ratio (Validation Loss / Training Loss)
fig, ax = plt.subplots(figsize=(10, 5))

loss_ratio_1 = val_loss_1 / (loss_1 + 1e-8)
loss_ratio_2 = val_loss_2 / (loss_2 + 1e-8)

ax.plot(epochs_1, loss_ratio_1, label="Pose Estimation", color="blue")
ax.plot(epochs_2, loss_ratio_2, label="Multimodal", color="red")
ax.set_title("Loss Ratio (Validation Loss / Training Loss)")
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss Ratio")
ax.legend()
ax.grid()

plt.savefig("output/loss_ratio.png")

# Standard Deviation of Accuracy & Loss Over Epochs
fig, axes = plt.subplots(figsize=(10, 8))

std_accuracy_1 = accuracy_1.expanding().std()
std_accuracy_2 = accuracy_2.expanding().std()

std_loss_1 = loss_1.expanding().std()
std_loss_2 = loss_2.expanding().std()

axes.plot(epochs_1, std_accuracy_1, label="Pose Estimation", color="blue")
axes.plot(epochs_2, std_accuracy_2, label="Multimodal", color="red")
axes.set_title("Standard Deviation of Accuracy Over Epochs")
axes.set_xlabel("Epochs")
axes.set_ylabel("Standard Deviation")
axes.legend()
axes.grid()
plt.savefig("output/accuracy_std.png")

axes.plot(epochs_1, std_loss_1, label="Pose Estimation", color="blue")
axes.plot(epochs_2, std_loss_2, label="Multimodal", color="red")
axes.set_title("Standard Deviation of Loss Over Epochs")
axes.set_xlabel("Epochs")
axes.set_ylabel("Standard Deviation")
axes.legend()
axes.grid()

plt.tight_layout()
plt.savefig("output/loss_std.png")

# Compute epoch-to-epoch improvement in accuracy
accuracy_improvement_1 = np.diff(accuracy_1, prepend=accuracy_1.iloc[0])
accuracy_improvement_2 = np.diff(accuracy_2, prepend=accuracy_2.iloc[0])

# Plot Accuracy Improvement Over Epochs
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(epochs_1, accuracy_improvement_1, label="Pose Estimation", color="blue")
ax.plot(epochs_2, accuracy_improvement_2, label="Multimodal", color="red")
ax.set_title("Epoch-wise Improvement in Accuracy")
ax.set_xlabel("Epochs")
ax.set_ylabel("Accuracy Improvement")
ax.axhline(0, color="black", linestyle="dashed")
ax.legend()
ax.grid()

plt.savefig("output/accuracy_improvement.png")

# Compute cumulative accuracy gain
cumulative_accuracy_gain_1 = np.cumsum(accuracy_improvement_1)
cumulative_accuracy_gain_2 = np.cumsum(accuracy_improvement_2)

# Plot Cumulative Accuracy Gain
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(epochs_1, cumulative_accuracy_gain_1, label="Pose Estimation", color="blue")
ax.plot(epochs_2, cumulative_accuracy_gain_2, label="Multimodal", color="red")
ax.set_title("Cumulative Accuracy Gain Over Epochs")
ax.set_xlabel("Epochs")
ax.set_ylabel("Cumulative Accuracy Gain")
ax.legend()
ax.grid()

plt.savefig("output/accuracy_gain.png")

# Simulated Confusion Matrix (Replace with real test predictions if available)
# Assuming both models classify three classes: Good (0), Bad (1), Uncertain (2)
true_labels = np.random.choice([0, 1, 2], size=100)  # Randomly generated test labels
predicted_labels_1 = np.random.choice([0, 1, 2], size=100)  # Simulated predictions for Pose Estimation
predicted_labels_2 = np.random.choice([0, 1, 2], size=100)  # Simulated predictions for Multimodal Approach

cm_1 = confusion_matrix(true_labels, predicted_labels_1)
cm_2 = confusion_matrix(true_labels, predicted_labels_2)

# Final accuracy and loss values
final_accuracy_pose = 0.9473
final_loss_pose = 0.1416

final_accuracy_multimodal = 0.6370
final_loss_multimodal = 0.6378

# Labels for approaches
approaches = ["Pose Estimation", "Multimodal"]

# Accuracy Bar Plot
plt.figure(figsize=(6, 5))
plt.bar(approaches, [final_accuracy_pose, final_accuracy_multimodal], color=["blue", "red"])
plt.title("Final Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)  # Accuracy ranges from 0 to 1
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig("output/final_accuracy_comparison.png")

# Loss Bar Plot
plt.figure(figsize=(6, 5))
plt.bar(approaches, [final_loss_pose, final_loss_multimodal], color=["blue", "red"])
plt.title("Final Loss Comparison")
plt.ylabel("Loss")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig("output/final_loss_comparison.png")

# Combined Accuracy & Loss Bar Plot
fig, ax1 = plt.subplots(figsize=(6, 5))

# Bar positions
x_pos = np.arange(len(approaches))

# Plot Accuracy (left y-axis)
ax1.bar(x_pos - 0.2, [final_accuracy_pose, final_accuracy_multimodal], width=0.4, label="Accuracy", color="blue")
ax1.set_ylabel("Accuracy", color="blue")
ax1.set_ylim(0, 1)  # Accuracy is between 0 and 1

# Create second y-axis for loss
ax2 = ax1.twinx()
ax2.bar(x_pos + 0.2, [final_loss_pose, final_loss_multimodal], width=0.4, label="Loss", color="red")
ax2.set_ylabel("Loss", color="red")

# Titles and labels
ax1.set_xticks(x_pos)
ax1.set_xticklabels(approaches)
ax1.set_title("Final Accuracy & Loss Comparison")

# Grid & Legend
ax1.grid(axis="y", linestyle="--", alpha=0.7)
fig.legend(loc="upper right", bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.savefig("output/final_accuracy_loss_comparison.png")


fig, axes = plt.subplots(figsize=(12, 5))

# Pose Estimation Confusion Matrix
sns.heatmap(cm_1, annot=True, fmt='d', cmap="Blues", xticklabels=["Good", "Bad", "Uncertain"], yticklabels=["Good", "Bad", "Uncertain"], ax=axes[0])
axes.set_title("Pose Estimation Confusion Matrix")
axes.set_xlabel("Predicted Label")
axes.set_ylabel("True Label")

# Multimodal Confusion Matrix
sns.heatmap(cm_2, annot=True, fmt='d', cmap="Reds", xticklabels=["Good", "Bad", "Uncertain"], yticklabels=["Good", "Bad", "Uncertain"], ax=axes[1])
axes.set_title("Multimodal Approach Confusion Matrix")
axes.set_xlabel("Predicted Label")
axes.set_ylabel("True Label")

plt.tight_layout()
plt.savefig("output/confusion_matrices.png")
