import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("dataset/employee_attrition.csv")

df.drop(
    ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"],
    axis=1,
    inplace=True
)

label_encoders = {}
categorical_cols = df.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

joblib.dump(model, "model.pkl")
joblib.dump(label_encoders, "encoders.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")

print("Training completed.")

print("Model saved to model.pkl")
print("Label encoders saved to encoders.pkl")
print("Model columns saved to model_columns.pkl")