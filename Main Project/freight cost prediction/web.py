from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)


# Load the trained model and encoders outside of the route functions
with open('Gradient_booster.pkl', 'rb') as file:
    best_gb_model = pickle.load(file)
    
    # Assign feature names to the model
    feature_names = ['Country', 'Managed By', 'Fulfill Via', 'Vendor INCO Term',
       'Shipment Mode', 'Sub Classification', 'Vendor', 'Brand', 'Dosage Form',
       'Line Item Value', 'First Line Designation', 'Weight (Kilograms)','Line Item Insurance (USD)', 'Delivery Status']
    
    
    best_gb_model.feature_names = feature_names
    
with open('label_encoders.pkl', 'rb') as file:
    label_encoders = pickle.load(file)
with open('robust_scaling.pkl', 'rb') as file:
    robust_scaling = pickle.load(file)    
    
    
# Define the columns used in preprocessing
label_cols = ['Country', 'Managed By', 'Fulfill Via', 'Vendor INCO Term','Shipment Mode', 'Sub Classification', 'Vendor', 'Brand', 'Dosage Form','First Line Designation','Delivery Status']
cols_to_scale = ['Line Item Value', 'Weight (Kilograms)','Line Item Insurance (USD)']


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Create a dictionary to store user input data
    input_data = {
        'Country': request.form.get('Country'),
        'Managed By': request.form.get('Managed By'),
        'Fulfill Via': request.form.get('Fulfill Via'),
        'Vendor INCO Term': request.form.get('Vendor INCO Term'),
        'Shipment Mode': request.form.get('Shipment Mode'),
        'Sub Classification': request.form.get('Sub Classification'),
        'Vendor': request.form.get('Vendor')    ,
        'Brand': request.form.get('Brand'),
        'Dosage Form': request.form.get('Dosage Form'),
        'First Line Designation': request.form.get('First Line Designation'),
        'Delivery Status': request.form.get('Delivery Status'),
        'Line Item Value': float(request.form.get('Line Item Value')),
        'Weight (Kilograms)': float(request.form.get('Weight (Kilograms)')),
        'Line Item Insurance (USD)': float(request.form.get('Line Item Insurance (USD)'))
    }
   

    # Create a DataFrame from the input data
    input_data_df = pd.DataFrame([input_data])
    
    
    # Label encoding
    encoded_label_cols = []
    for col, le in label_encoders.items():
        encoded_col = le.transform(input_data_df[col])
        encoded_label_cols.append(encoded_col)

    encoded_label = np.column_stack(encoded_label_cols)

      
    # Robust scaling for specified columns
    scaled_cols = []
    for col, rs in robust_scaling.items():
        scaled_col = rs.transform(input_data_df[[col]])
        scaled_cols.append(scaled_col)

    scaled = np.column_stack(scaled_cols)
    
    
    # Concatenate all parts together
    encoded_input = np.concatenate((encoded_label, scaled), axis=1)
    # Create a DataFrame with the encoded input data and feature names
    encoded_input_df = pd.DataFrame(encoded_input, columns=feature_names)
    
    
    # Make a prediction using the loaded model
    prediction = best_gb_model.predict(encoded_input_df)
    
    # Render the prediction result on a new page or template
    return render_template('result.html', prediction_text="Predicted Freight Cost: ${:.2f}".format(prediction[0]))

if __name__ == '__main__':
    app.run(debug=False, port=5067)