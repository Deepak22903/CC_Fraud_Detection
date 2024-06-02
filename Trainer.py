import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pickle

class FraudDetectionModel:
    def __init__(self, data_path, nrows=None):
        """
        Initialize the FraudDetectionModel object.
        
        Args:
        - data_path (str): Path to the dataset.
        - nrows (int, optional): Number of rows to read from the dataset. Default is None (read all rows).
        """
        self.data_path = data_path
        self.nrows = nrows
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_smote = None
        self.y_train_smote = None
        self.models = []

    def load_data(self):
        """
        Load the dataset from the specified path.
        """
        self.data = pd.read_csv(self.data_path, nrows=self.nrows)

    def preprocess_data(self):
        """
        Preprocess the loaded dataset.
        """
        scaler = StandardScaler()
        self.data['Amount_Scaled'] = scaler.fit_transform(self.data['Amount'].values.reshape(-1, 1))
        self.data = self.data.drop(['Time', 'Amount'], axis=1)

    def split_data(self, test_size=0.2, random_state=42):
        """
        Split the dataset into training and testing sets.
        
        Args:
        - test_size (float, optional): The proportion of the dataset to include in the test split. Default is 0.2.
        - random_state (int, optional): Random seed for reproducibility. Default is 42.
        """
        X = self.data.drop('Class', axis=1).values
        y = self.data['Class'].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    def balance_classes(self, k_neighbors=1, random_state=42):
        """
        Balance the classes in the training data using SMOTE.
        
        """
        smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        self.X_train_smote, self.y_train_smote = smote.fit_resample(self.X_train, self.y_train)

    def train_models(self):
        """
        Train machine learning models on the balanced training data.
        """
        logreg = LogisticRegression(max_iter=1000000)
        rfc = RandomForestClassifier(n_estimators=100, random_state=42)
        xgb = XGBClassifier(n_estimators=100, random_state=42)
        
        self.models = [('Logistic Regression', logreg), ('Random Forest', rfc), ('XGBoost', xgb)]
        
        for name, model in self.models:
            model.fit(self.X_train_smote, self.y_train_smote)

    def evaluate_models(self):
        """
        Evaluate the performance of trained models on the test data.
        """
        for name, model in self.models:
            y_pred = model.predict(self.X_test)
            y_prob = model.predict_proba(self.X_test)[:, 1]
            print(f'{name}:')
            print(f'Accuracy: {accuracy_score(self.y_test, y_pred):.4f}')
            print(f'Precision: {precision_score(self.y_test, y_pred):.4f}')
            print(f'Recall: {recall_score(self.y_test, y_pred):.4f}')
            print(f'F1-score: {f1_score(self.y_test, y_pred):.4f}')
            print(f'AUC-ROC score: {roc_auc_score(self.y_test, y_prob):.4f}')
            print(f'Confusion Matrix: \n{confusion_matrix(self.y_test, y_pred)}\n')

    def save_model(self, filename='fraud_detection_model.sav'):
        """
        Save the trained XGBoost model.
        
        Args:
        - filename (str, optional): Name of the file to save the model. Default is 'fraud_detection_model.sav'.
        """
        pickle.dump(self.models[-1][1], open(filename, 'wb'))


# Example usage:
if __name__ == "__main__":
    model = FraudDetectionModel("creditcard.csv", nrows=100000)
    model.load_data()
    model.preprocess_data()
    model.split_data()
    model.balance_classes()
    model.train_models()
    model.evaluate_models()
    model.save_model()
