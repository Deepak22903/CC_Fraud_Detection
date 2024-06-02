import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy import stats
import pickle

# Load the dataset
df = pd.read_csv("fraudTrain.csv", nrows=10000)

# Create the feature matrix and target vector
X = df.drop(["is_fraud"], axis=1)
y = df["is_fraud"]

# Convert categorical variables to dummy variables
X = pd.get_dummies(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train individual models
models = []
model_names = ["Random Forest", "AdaBoost", "Logistic Regression"]
model_weights = [1, 1, 1]  # Adjust the weights for voting

# Train Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
models.append(rf)

# Train AdaBoost model
adb = AdaBoostClassifier(n_estimators=100, random_state=42)
adb.fit(X_train, y_train)
models.append(adb)

# Train Logistic Regression model
lr = LogisticRegression(random_state=42, max_iter=500)
lr.fit(X_train, y_train)
models.append(lr)

# Combine predictions using weighted voting
weighted_predictions = None

for model, weight in zip(models, model_weights):
    predictions = model.predict(X_test)
    if weighted_predictions is None:
        weighted_predictions = weight * predictions
    else:
        weighted_predictions += weight * predictions

weighted_predictions = weighted_predictions.astype(float)  # Convert to float before division
weighted_predictions /= sum(model_weights)

weighted_predictions = weighted_predictions.round().astype(int)

# Combine predictions using majority voting
majority_predictions = stats.mode([model.predict(X_test) for model in models], axis=0).mode[0]

# Evaluate the combined models
print("Weighted Voting - Confusion Matrix:\n", confusion_matrix(y_test, weighted_predictions))
print("\nWeighted Voting - Classification Report:\n", classification_report(y_test, weighted_predictions))
print("\nMajority Voting - Confusion Matrix:\n", confusion_matrix(y_test, majority_predictions))
print("\nMajority Voting - Classification Report:\n", classification_report(y_test, majority_predictions))

# Saving the trained models
filename = 'fraud_detection_model3.sav'
pickle.dump(models, open(filename, 'wb'))

# Rest of the code remains the same...
