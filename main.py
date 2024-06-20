import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
    
# Load Data
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/BostonHousing.csv')

"""     Column Headings
CRIM - per capita crime rate by town
ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
INDUS - proportion of non-retail business acres per town.
CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
NOX - nitric oxides concentration (parts per 10 million)
RM - average number of rooms per dwelling
AGE - proportion of owner-occupied units built prior to 1940
DIS - weighted distances to five Boston employment centres
RAD - index of accessibility to radial highways
TAX - full-value property-tax rate per $10,000
PTRATIO - pupil-teacher ratio by town
B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
LSTAT - % lower status of the population
MEDV - Median value of owner-occupied homes in $1000's
"""
# Data Preparation

# Data separation as X and Y (dependent variable)
price = df['medv']
variables = df.drop('medv', axis=1)

# Data Splitting: 80% Training, 20% Testing
variables_train, variables_test, price_train, price_test = \
train_test_split(variables, price, test_size=0.2, random_state=100)

# Model Building

## Linear Regression

### Training the Model
lr = LinearRegression()
lr.fit(variables_train, price_train)

### Applying the model to make a prediction

price_lr_train_pred = lr.predict(variables_train)
price_lr_test_pred = lr.predict(variables_test)

### Evaluate Model Performance
lr_train_mse = mean_squared_error(price_train, price_lr_train_pred)
lr_train_r2 = r2_score(price_train, price_lr_train_pred)

lr_test_mse = mean_squared_error(price_test, price_lr_test_pred)
lr_test_r2 = r2_score(price_test, price_lr_test_pred)

lr_results = pd.DataFrame(['Linear Regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']

## Random Forest
### Training the model
rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(variables_train, price_train)

### Applying the model to make predictions
price_rf_train_pred = rf.predict(variables_train)
price_rf_test_pred = rf.predict(variables_test)

### Evaluate Model Performance
rf_train_mse = mean_squared_error(price_train, price_rf_train_pred)
rf_train_r2 = r2_score(price_train, price_rf_train_pred)

rf_test_mse = mean_squared_error(price_test, price_rf_test_pred)
rf_test_r2 = r2_score(price_test, price_rf_test_pred)

rf_results = pd.DataFrame(['Random Forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']


## Model Comparison
df_models = pd.concat([lr_results, rf_results], axis=0).reset_index(drop=True)



def predict_price(features):
    input_data = pd.DataFrame([features], columns=variables.columns)
    prediction_lr = lr.predict(input_data)[0]
    prediction_rf = rf.predict(input_data)[0]
    prediction_avg = (prediction_lr + prediction_rf) / 2
    return prediction_lr, prediction_rf, prediction_avg

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction_lr, prediction_rf, prediction_avg = predict_price(features)
    return jsonify({
        'prediction_lr': prediction_lr * 1000,  # convert to dollars
        'prediction_rf': prediction_rf * 1000,  # convert to dollars
        'prediction_avg': prediction_avg * 1000 # convert to dollars
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        data = pd.read_csv(file_path)
        predictions_lr = []
        predictions_rf = []
        predictions_avg = []
        for _, row in data.iterrows():
            features = row.values.tolist()
            prediction_lr, prediction_rf, prediction_avg = predict_price(features)
            predictions_lr.append(prediction_lr * 1000)  # convert to dollars
            predictions_rf.append(prediction_rf * 1000)  # convert to dollars
            predictions_avg.append(prediction_avg * 1000) # convert to dollars
        
        return jsonify({
            'linear_regression': predictions_lr,
            'random_forest': predictions_rf,
            'average_prediction': predictions_avg
        })

if __name__ == "__main__":
    app.run(debug=True)