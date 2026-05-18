import pandas as pd
import matplotlib.pyplot as plt

# Load grouped dataset
df = pd.read_excel(r"C:\Users\ASUS\OneDrive\Documents\GitHub\2-BSCS-A-Final_Computational_Science_Project\disaster\grouped_dataset.xlsx")

# Count samples per group
class_counts = df["Grouped Label"].value_counts().sort_index()

# Map numeric labels to names
labels = {1: "Hydrological", 2: "Meteorological", 3: "Geological"}
class_counts.index = class_counts.index.map(labels)

# Plot bar chart
plt.figure(figsize=(7,5))
bars = plt.bar(class_counts.index, class_counts.values, color=['blue','green','orange'])
plt.ylabel("Number of Samples")
plt.title("Class Distribution of Grouped Labels")

# Add counts on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 2, str(height),
             ha='center', va='bottom', fontsize=10)

plt.show()
