# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Load the dataset with specified encoding
df = pd.read_csv("D:/CLavens mern project/archive (3)/Sample - Superstore.csv", encoding='latin1')

# Step 3: Clean the data (Optional based on your dataset)
# Check for missing values and handle them (e.g., drop or fill missing values)
df = df.dropna(subset=['Profit', 'Sales'])

# Step 4: Feature Engineering
# Set a threshold for profit to define high vs low sales
threshold = df['Profit'].median()  # You can adjust this threshold as needed
df['Sales_Label'] = np.where(df['Profit'] > threshold, 1, 0)  # 1 for High sales, 0 for Low sales

# Step 5: One-Hot Encoding for categorical columns
df = pd.get_dummies(df, columns=['Segment', 'Country', 'City', 'State', 'Category'], drop_first=True)

# Step 6: Select features (use Profit, Sales, and other relevant features)
X = df[['Profit', 'Sales', 'Quantity'] + [col for col in df.columns if col.startswith(('Segment_', 'Country_', 'City_', 'State_', 'Category_'))]]  # Features used for classification
y = df['Sales_Label']  # Target variable (High/Low sales)

# Step 7: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train the Logistic Regression model
model = LogisticRegression(max_iter=10000)  # Increased max_iter for better convergence
model.fit(X_train, y_train)

# Step 9: Predict on the test set
y_pred = model.predict(X_test)

# Step 10: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)



# Step 11: JSON Test Data for prediction

# Single JSON data (same structure as the model)
test_data_json = '''
{
  "Profit": 41.9,
  "Sales": 261.96,
  "Quantity": 2,
  "Segment": "Consumer",
  "Country": "United States",
  "City": "Henderson",
  "State": "Kentucky",
  "Category": "Furniture"
}
'''

# Step 12: Convert the JSON string to a Python object
test_data = json.loads(test_data_json)

# Step 13: Convert the test data into a DataFrame (same structure as the model)
test_df = pd.DataFrame([test_data])

# Step 14: Apply the same one-hot encoding to the test data
test_df = pd.get_dummies(test_df, columns=['Segment', 'Country', 'City', 'State', 'Category'], drop_first=True)

# Step 14.1: Add any missing columns from the training set
missing_cols = set(X.columns) - set(test_df.columns)
for col in missing_cols:
    test_df[col] = 0  # Add missing columns with default value 0

# Step 14.2: Reorder the test columns to match the training data
test_df = test_df[X.columns]

# Step 16: Predict on the test data using the trained model
y_pred_test = model.predict(test_df)

# Step 17: Show the predictions
print("\nPrediction for test data:", y_pred_test)
print("Profit Median Threshold:", threshold)

