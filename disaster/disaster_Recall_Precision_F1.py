import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Load grouped dataset
df = pd.read_excel(r"C:\Users\ASUS\OneDrive\Documents\GitHub\2-BSCS-A-Final_Computational_Science_Project\disaster\grouped_dataset.xlsx")

# Features and labels
X = df.drop(["Disaster Label","Grouped Label"], axis=1)
y = df["Grouped Label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Naïve Bayes ---
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
print("Naïve Bayes Classification Report:\n")
print(classification_report(y_test, y_pred_nb, digits=3))

# --- KNN (best k=7) ---
knn_model = KNeighborsClassifier(n_neighbors=7)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
print("KNN Classification Report (k=7):\n")
print(classification_report(y_test, y_pred_knn, digits=3))

# --- Decision Tree ---
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
print("Decision Tree Classification Report:\n")
print(classification_report(y_test, y_pred_dt, digits=3))
