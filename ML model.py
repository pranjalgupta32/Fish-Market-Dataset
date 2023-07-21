import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Step 1: Load the dataset
url = "https://www.kaggle.com/aungpyaeap/fish-market"
# Replace the above 'url' variable with the path to the local file if you have downloaded the dataset.

# Assuming the dataset is saved as 'Fish.csv'
df = pd.read_csv("Fish.csv")

# Step 2: Data Preprocessing
# Check for any missing values and handle them if present (not necessary for this dataset).
# Drop the 'Species' column from the features, as it is the target variable.
X = df.drop(columns=['Species'])
y = df['Species']

# Encode the categorical target variable 'Species' into numerical labels.
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Split the dataset into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Build the Machine Learning model
rf_model = RandomForestClassifier(random_state=42)

# Fit the model to the training data
rf_model.fit(X_train, y_train)

# Step 4: Evaluate the model
y_pred = rf_model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate the classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Step 5: Visualize the results (optional)
# For instance, you can create a bar plot to show feature importances if you want to analyze feature importance.
feature_importances = rf_model.feature_importances_
feature_names = X.columns

plt.barh(feature_names, feature_importances)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importances in Random Forest')
plt.show()
