import pandas as pd
import os
import pickle

file_path = "/home/seabata/Documents/Data_science_project/customer_churn_dataset_testing_master.csv"

if os.path.exists(file_path):
    data = pd.read_csv(file_path)
    print("File loaded successfully!")
else:
    print(f"Error: The file {file_path} was not found. Please check the file path.")

print(data.head())
print(data.info())

#LbelEncoding-assigns unique integer to categorical vars
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])  # Example for a column named 'Gender'
data['Subscription Type'] = le.fit_transform(data['Subscription Type'])  # Example for a column named 'Subscription Type'
data['Contract Length'] = le.fit_transform(data['Contract Length'])  # Example for a column named 'Contract Length'

#One HOtEncoding-creates new binary columns for each category
data = pd.get_dummies(data, columns=['Gender','Subscription Type','Contract Length'], drop_first=True)  # Example for a column named 'Gender'
print(data.isnull().sum()) #check for missing values in our data
data.fillna(0, inplace=True) #Replace missing values with 0
#recalculate the corr matrix
corr_matrix = data.corr()
print(corr_matrix)

#Visualize the corr matrix
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Splitting the data
X = data.drop('Churn', axis=1)
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


# Assuming your trained model is saved in the variable `model`
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Code done running")





