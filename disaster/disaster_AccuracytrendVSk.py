import matplotlib.pyplot as plt

# Accuracy values from grouped dataset
k_values = [3, 5, 7]
accuracies = [49.0, 49.0, 52.33]

plt.figure(figsize=(8,5))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='blue')

plt.title("Accuracy Trend vs. K (KNN - Grouped Dataset)")
plt.xlabel("K Value")
plt.ylabel("Accuracy (%)")
plt.xticks(k_values)
plt.ylim(45, 55)  # adjust range for clarity
plt.grid(True)

# Add labels on each point
for k, acc in zip(k_values, accuracies):
    plt.text(k, acc+0.3, f"{acc:.2f}%", ha='center')

plt.show()
