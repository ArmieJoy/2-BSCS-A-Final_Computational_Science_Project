import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load grouped dataset
df = pd.read_excel(r"C:\Users\ASUS\OneDrive\Documents\GitHub\2-BSCS-A-Final_Computational_Science_Project\disaster\grouped_dataset.xlsx")

# Features and labels
X = df.drop(["Disaster Label","Grouped Label"], axis=1)
y = df["Grouped Label"]

# Train/test split (stratify keeps balance across 3 groups)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    "Naïve Bayes": GaussianNB(),
    "KNN (k=3)": KNeighborsClassifier(n_neighbors=3),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "KNN (k=7)": KNeighborsClassifier(n_neighbors=7),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# Train and evaluate
accuracies = {}
for name, model in models.items():
    if "KNN" in name:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    accuracies[name] = round(accuracy_score(y_test, y_pred) * 100, 2)  # two decimals

# Results table
comparison_df = pd.DataFrame(list(accuracies.items()), columns=["Algorithm", "Accuracy (%)"])
print(comparison_df)

# Bar graph
plt.figure(figsize=(8,6))
plt.bar(comparison_df["Algorithm"], comparison_df["Accuracy (%)"], 
        color=['blue','green','orange','red','purple'])
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Comparison of Algorithms (Grouped Labels)")
plt.ylim(0,100)

# Add labels on top of bars
for i, v in enumerate(comparison_df["Accuracy (%)"]):
    plt.text(i, v + 1, str(v) + "%", ha='center', fontsize=9)

plt.show()
