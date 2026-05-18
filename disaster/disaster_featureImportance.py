import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load grouped dataset
df = pd.read_excel(r"C:\Users\ASUS\OneDrive\Documents\GitHub\2-BSCS-A-Final_Computational_Science_Project\disaster\grouped_dataset.xlsx")

# Features and labels
X = df.drop(["Disaster Label","Grouped Label"], axis=1)
y = df["Grouped Label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Get feature importance
importances = dt_model.feature_importances_
features = X.columns

# Create a color palette (different color per feature)
colors = plt.cm.tab20(np.linspace(0, 1, len(features)))

# Plot feature importance
plt.figure(figsize=(12,7))
bars = plt.barh(features, importances, color=colors)
plt.xlabel("Importance Score")
plt.title("Feature Importance - Decision Tree")

# Add values with 3 decimal places
for bar, imp in zip(bars, importances):
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
             f"{imp:.3f}", va='center', fontsize=9)

plt.tight_layout()
plt.show()
