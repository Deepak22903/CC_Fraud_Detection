# Import required libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle

# Load the dataset
df = pd.read_csv("fraudTrain.csv", nrows=10000)

# Create the feature matrix and target vector%t
X = df.drop(["is_fraud"], axis=1)
y = df["is_fraud"]

# Convert categorical variables to dummy variables
# Before converting categorical variables to dummy variables
print("X shape before:", X.shape)

# Convert categorical variables to dummy variables
X = pd.get_dummies(X)

# After converting categorical variables to dummy variables
print("X shape after:", X.shape)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model using Random Forest algorithm
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict on the test set
y_pred = rf.predict(X_test)

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Saving the trained model
filename = 'fraud_detection_model2.sav'
pickle.dump(rf, open(filename, 'wb'))

# Read the specific row at index 2450
row_index = 2450
new_data = pd.read_csv("fraudTrain.csv", skiprows=range(0, row_index), nrows=1)

# Verify the features
training_features = X.columns
new_data_features = new_data.columns

print("Features:", new_data_features)

# Handle missing features
missing_features = list(set(training_features) - set(new_data_features))
missing_data = pd.DataFrame(0, index=new_data.index, columns=missing_features)
new_data = pd.concat([new_data, missing_data], axis=1)

# Reorder the features to match the training data
new_data = new_data[training_features]


# Convert categorical variables to dummy variables
new_data = pd.get_dummies(new_data)

# Load the trained model
loaded_model = pickle.load(open(filename, 'rb'))

# Make predictions
predictions = loaded_model.predict(new_data)

if predictions[0] == 0:
    print("The transaction is not fraudulent.")
else:
    print("The transaction is fraudulent!")

# Train the model using Random Forest algorithm
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
importances = rf.feature_importances_

# Create a DataFrame to display feature importances
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importances = feature_importances.sort_values('Importance', ascending=False)

# Print the most important features
print("Most important features:")
print(feature_importances.head(10))  # Adjust the number (e.g., 10) as needed
