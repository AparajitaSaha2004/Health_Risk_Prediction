import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv("diabetes.csv")

# Convert Yes/No columns
for column in df.columns:
    if column not in ["Age", "Gender", "class"]:
        df[column] = df[column].map({"Yes": 1, "No": 0})

# Convert Gender
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

# Convert target
df["class"] = df["class"].map({"Positive": 1, "Negative": 0})

print("\nConverted data:")
print(df.head())

print("\nData types:")
print(df.dtypes)

# Separate features and target
X = df.drop("class", axis=1)
y = df["class"]

print("\nX shape:", X.shape)
print("y shape:", y.shape)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# Create the model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

print("\nModel trained successfully!")

# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, "diabetes_model.pkl")

print("\nModel saved successfully!")