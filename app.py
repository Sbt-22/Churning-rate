from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the saved model
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Extract input data from form
    Payment_Delay = float (request.form['Payment_Delay'])
    Age = float (request.form['Age'])
    Tenure = float (request.form['Tenure'])
    Support_calls = float (request.form['Support_calls'])
    features_array = np.array([Payment_Delay, Age, Tenure, Support_calls])
    
    # Make a prediction using the loaded model
    prediction = model.predict(features_array)
    
    # Convert prediction to readable format
    output = "Churn" if prediction[0] == 1 else "No Churn"
    
    return render_template('index.html', prediction_text=f'Customer Status: {output}')

if __name__ == "__main__":
    app.run(debug=True)

