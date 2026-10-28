import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load data
df = pd.read_csv("../data/garba.csv")
print(f"Total frames collected: {len(df)}")

# Remove missing data
df = df.dropna()
print(f"After cleaning: {len(df)}")

# Separate features and labels
label = df["label"]
X = df.drop(columns=["label"]).values

# Normalize pose by hip center
def normalize_pose(pose):
    pose = pose.reshape(33, -1)
    hip_center = (pose[23, :2] + pose[24, :2]) / 2
    pose[:, :2] = pose[:, :2] - hip_center
    shoulder_dist = np.linalg.norm(pose[11, :2] - pose[12, :2])
    if shoulder_dist > 0:
        pose[:, :2] = pose[:, :2] / shoulder_dist
    return pose.flatten()

X_normalized = np.array([normalize_pose(x) for x in X])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, label, test_size=0.2, stratify=label, random_state=42
)

print(f"\nTraining set: {len(X_train)}, Test set: {len(X_test)}")

# Train Random Forest
print("\nTraining model...")
model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✓ Accuracy: {accuracy:.2%}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(model, "garba_model.pkl")
print("\n✓ Model saved as garba_model.pkl")

# Save reference poses (average per step)
reference_poses = {}
for step in df["label"].unique():
    step_data = df[df["label"] == step].drop(columns=["label"]).values
    avg_pose = step_data.mean(axis=0)
    reference_poses[step] = normalize_pose(avg_pose)

joblib.dump(reference_poses, "reference_poses.pkl")
print("✓ Reference poses saved as reference_poses.pkl")