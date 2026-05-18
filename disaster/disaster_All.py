import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import tkinter as tk
from tkinter import simpledialog, messagebox

# -----------------------------
# 1. Load Dataset
# -----------------------------
#File save to a folder (disaster) in desktop
data = pd.read_excel(r"C:\Users\ASUS\OneDrive\Documents\GitHub\2-BSCS-A-Final_Computational_Science_Project\disaster\grouped_dataset.xlsx")

# Drop both label columns (targets)
X = data.drop(["Disaster Label", "Grouped Label"], axis=1)
y = data["Disaster Label"]   # Use Disaster Label as target

# -----------------------------
# 2. Preprocessing
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 3. Train Models
# -----------------------------
nb_model = GaussianNB().fit(X_train, y_train)
knn_model = KNeighborsClassifier(n_neighbors=3).fit(X_train_scaled, y_train)
dt_model = DecisionTreeClassifier(criterion="entropy", random_state=42).fit(X_train, y_train)

# -----------------------------
# 4. GUI Input (Survey Questions)
# -----------------------------
root = tk.Tk()
root.withdraw()

messagebox.showinfo("Disaster Prediction", "Please answer the following survey questions (type the numeric code).")

# KNN Features
rainfall_intensity = int(simpledialog.askstring("Input", "During heavy rainfall, how would you describe the rainfall intensity in your area? (1=Very Light, 2=Light, 3=Moderate, 4=Heavy, 5=Very Heavy)"))
water_level = int(simpledialog.askstring("Input", "How high does flood water usually rise in your area? (1=Below ankle, 2=Knee, 3=Waist, 4=Chest, 5=Above chest)"))
wind_strength = int(simpledialog.askstring("Input", "During storms or typhoons, how strong is the wind usually experienced? (1=Weak, 2=Moderate, 3=Strong, 4=Very Strong)"))
flood_risk = int(simpledialog.askstring("Input", "Based on your experience, what is the flood risk level in your area? (1=Low, 2=Medium, 3=High)"))

# Decision Tree Features
rainfall_disaster = int(simpledialog.askstring("Input", "How would you classify rainfall during disaster events in your area? (1=Light, 2=Moderate, 3=Heavy)"))
river_overflow = int(simpledialog.askstring("Input", "Does river overflow usually occur during heavy rain? (0=No, 1=Yes)"))
soil_condition = int(simpledialog.askstring("Input", "What is the condition of the soil in your area during heavy rain? (1=Weak, 2=Stable)"))
wind_intensity = int(simpledialog.askstring("Input", "How would you describe wind intensity during disasters? (1=Weak, 2=Moderate, 3=Strong)"))
disaster_severity = int(simpledialog.askstring("Input", "Based on your observation, how severe are disasters in your area? (1=Low, 2=Moderate, 3=Severe)"))

# Naïve Bayes Features
cloud_density = int(simpledialog.askstring("Input", "Before a typhoon occurs, how would you describe cloud density? (1=Low, 2=Medium, 3=High)"))
humidity = int(simpledialog.askstring("Input", "How would you describe humidity before heavy rain or typhoon? (1=Low, 2=Medium, 3=High)"))
rainfall_typhoon = int(simpledialog.askstring("Input", "How would you classify rainfall during typhoon events? (1=Light, 2=Moderate, 3=Heavy)"))
wind_speed = int(simpledialog.askstring("Input", "How would you classify wind speed during typhoons? (1=Weak, 2=Moderate, 3=Strong)"))
typhoon_warning = int(simpledialog.askstring("Input", "Based on weather conditions, do you think a typhoon warning should be issued? (0=No, 1=Yes)"))

# -----------------------------
# 5. Predictions with Full Probabilities
# -----------------------------
user_input = [[rainfall_intensity, water_level, wind_strength, flood_risk,
               rainfall_disaster, river_overflow, soil_condition, wind_intensity, disaster_severity,
               cloud_density, humidity, rainfall_typhoon, wind_speed, typhoon_warning]]

# Scale for KNN
user_input_scaled = scaler.transform(user_input)

# Probabilities for each class
nb_probs = nb_model.predict_proba(user_input)[0]
knn_probs = knn_model.predict_proba(user_input_scaled)[0]
dt_probs = dt_model.predict_proba(user_input)[0]

# -----------------------------
# 6. Map numeric codes to disaster names
# -----------------------------
disaster_map = {
    0: "Flood",
    1: "Typhoon",
    2: "Landslide",
    3: "Earthquake",
    4: "Storm Surge",
    5: "Fire",
    6: "Drought"
}

# -----------------------------
# 7. Format Results with Percentages
# -----------------------------
def format_probs(probs, model_name):
    lines = [f"{model_name} Probabilities:"]
    for i, p in enumerate(probs):
        lines.append(f"  {disaster_map[i]} → {p*100:.2f}%")
    # Highlight the most confident prediction
    best_idx = probs.argmax()
    lines.append(f"Most Confident Prediction → {disaster_map[best_idx]} ({probs[best_idx]*100:.2f}%)")
    return "\n".join(lines)

result_message = f"""
=== Prediction Results (Full Breakdown) ===

{format_probs(nb_probs, "Naïve Bayes")}

{format_probs(knn_probs, "KNN (k=3)")}

{format_probs(dt_probs, "Decision Tree")}
"""

# Show the full breakdown first
messagebox.showinfo("Prediction Results", result_message)

# -----------------------------
# 8. Final Consensus (Most Confident Overall)
# -----------------------------
# Collect all top predictions with their confidence
all_predictions = [
    ("Naïve Bayes", disaster_map[nb_probs.argmax()], nb_probs.max()),
    ("KNN", disaster_map[knn_probs.argmax()], knn_probs.max()),
    ("Decision Tree", disaster_map[dt_probs.argmax()], dt_probs.max())
]

# Find the single most confident prediction across all models
final_model, final_label, final_conf = max(all_predictions, key=lambda x: x[2])

# Show final popup
messagebox.showinfo("Final Prediction", f"{final_label} ({final_conf*100:.2f}% confidence from {final_model})")

